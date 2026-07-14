from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

import audit_causal_noop_outputs as auditor
import run_causal_noop_experiment as runner
from sink_neurons.attention_noop import (
    InterventionSite,
    SiteResult,
    bootstrap_sequence_summaries,
    choose_target_positions,
    identify_sink_heads,
    kl_clean_intervened,
    redirect_attention_mass,
    run_attention_site,
    run_attention_sites_batch,
    run_secondary_sequence,
    run_value_site,
    summarize_sequence_pairs,
)


def check_core_math() -> None:
    old_argv = sys.argv
    sys.argv = ["run_causal_noop_experiment.py"]
    try:
        args = runner.parse_args()
    finally:
        sys.argv = old_argv
    assert args.datasets == ["wikitext_test"]
    assert args.eval_windows == 256
    assert args.primary_delta == 0.05
    assert args.sensitivity_deltas == [0.02, 0.10]
    assert args.queries_per_sequence == 1
    assert args.site_batch_size > 1


    attn = torch.tensor([[0.2, 0.3, 0.5], [0.95, 0.03, 0.02]], dtype=torch.float32)
    rewritten, skipped = redirect_attention_mass(attn, target_position=1, delta=0.1)
    assert torch.allclose(rewritten[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(rewritten[0, 1], attn[0, 1] + 0.1)
    assert not bool(skipped[0])

    rewritten2, skipped2 = redirect_attention_mass(attn, target_position=0, delta=0.1)
    assert bool(skipped2[1])
    assert torch.allclose(rewritten2[1], attn[1])

    ids = torch.tensor([0, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    targets = choose_target_positions(ids, 8, sink_position=0, special_ids={0}, rng=random.Random(0))
    assert targets is not None
    assert targets["sink"] == 0
    assert 1 <= targets["early"] <= 4

    kl = kl_clean_intervened(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(kl, torch.tensor([0.0]), atol=1e-7)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [1, 2]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return {"input_ids": [3 + (ord(ch) % 50) for ch in text if not ch.isspace()]}


def check_fixed_pile_text_windows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        text_artifact = tmp_path / "fixed_pile_texts.json"
        texts = ["alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda mu"]
        text_artifact.write_text('{"texts": ' + __import__("json").dumps(texts) + '}\n')

        loaded_texts, loaded_path = runner.load_or_build_pile_texts(
            out=tmp_path,
            path=text_artifact,
            dataset_name="unused",
            dataset_config=None,
            split="validation",
            count=3,
            show_progress=False,
        )
        assert loaded_path == text_artifact
        assert loaded_texts == texts

        original_load_tokenizer = runner.load_tokenizer
        runner.load_tokenizer = lambda _model_id: FakeTokenizer()  # type: ignore[assignment]
        try:
            windows_a = runner.tokenize_pile_texts_to_windows(
                out=tmp_path,
                model_key="model_a",
                model_id="fake-a",
                texts=loaded_texts,
                text_artifact=loaded_path,
                count=2,
                length=8,
                stride=4,
                show_progress=False,
            )
            windows_b = runner.tokenize_pile_texts_to_windows(
                out=tmp_path,
                model_key="model_b",
                model_id="fake-b",
                texts=loaded_texts,
                text_artifact=loaded_path,
                count=2,
                length=8,
                stride=4,
                show_progress=False,
            )
        finally:
            runner.load_tokenizer = original_load_tokenizer  # type: ignore[assignment]

        assert windows_a.shape == (2, 8)
        assert windows_b.shape == (2, 8)
        assert torch.equal(windows_a, windows_b)
        assert int(windows_a[0, 0]) == FakeTokenizer.bos_token_id


def check_sharded_eval_helpers() -> None:
    windows = torch.arange(60, dtype=torch.long).view(5, 12)
    sliced = runner.slice_eval_windows(windows, start=2, count=2)
    assert torch.equal(sliced, windows[2:4])

    selection = runner.HeadSelection(
        sink_position=0,
        sink_heavy=(runner.HeadRef(layer=0, head=0, sink_strength=0.9),),
        low_sink=(runner.HeadRef(layer=0, head=1, sink_strength=0.1),),
        all_scores=(),
    )
    shard_windows = torch.tensor([
        [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        [1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
    ])
    sites = runner.build_sites(
        windows=shard_windows,
        selection=selection,
        tokenizer=FakeTokenizer(),
        deltas=[0.02],
        queries_per_sequence=1,
        query_start=16,
        seed=0,
        sequence_offset=10,
    )
    assert sites
    assert {site.sequence_index for site in sites} == {10, 11}
    assert {site.target_kind for site in sites} == {"sink", "early", "random"}
    assert len({(site.sequence_index, site.head_kind, site.layer, site.head) for site in sites if site.head_kind == "sink_heavy"}) == 2



def check_sequence_statistics() -> None:
    rows: list[SiteResult] = []
    for layer, sink_kl, control_kl in [(0, 0.1, 0.3), (4, 0.2, 0.25), (8, 0.4, 0.1)]:
        rows.append(SiteResult(0, 16, layer, 0, "sink_heavy", "sink", 0, 0.05, sink_kl, 0, 0, 0, False))
        rows.append(SiteResult(0, 16, layer, 0, "sink_heavy", "early", 1, 0.05, control_kl, 0, 0, 0, False))
    summaries = summarize_sequence_pairs(rows)
    assert {row.layer_band for row in summaries} >= {"all", "early", "middle", "late"}
    bootstrap = bootstrap_sequence_summaries(summaries, samples=20)
    assert {row.layer_band for row in bootstrap} >= {"all", "early", "middle", "late"}

    multi_query_rows = [
        SiteResult(0, 16, 0, 0, "sink_heavy", "sink", 0, 0.05, 0.10, 0.01, 0.4, -0.01, False),
        SiteResult(0, 16, 0, 0, "sink_heavy", "early", 1, 0.05, 0.30, 0.04, 1.0, -0.05, False),
        SiteResult(0, 17, 0, 0, "sink_heavy", "sink", 0, 0.05, 0.20, 0.02, 0.6, -0.02, False),
        SiteResult(0, 17, 0, 0, "sink_heavy", "early", 2, 0.05, 0.60, 0.07, 1.6, -0.08, False),
    ]
    multi_query_summary = [
        row for row in summarize_sequence_pairs(multi_query_rows)
        if row.layer_band == "all" and row.control_kind == "early"
    ]
    assert len(multi_query_summary) == 1
    assert multi_query_summary[0].num_pairs == 2
    assert abs(multi_query_summary[0].mean_control_minus_sink_kl - 0.30) < 1e-9
    assert abs(multi_query_summary[0].mean_control_minus_sink_loss - 0.04) < 1e-9
    assert abs(multi_query_summary[0].mean_control_minus_sink_logit_l2 - 0.8) < 1e-9
    assert abs(multi_query_summary[0].mean_control_minus_sink_correct_prob - (-0.05)) < 1e-9
    multi_query_bootstrap = bootstrap_sequence_summaries(multi_query_summary, samples=20)
    assert abs(multi_query_bootstrap[0].mean_control_minus_sink_loss - 0.04) < 1e-9


def _tiny_architectures():
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    yield (
        "llama",
        LlamaConfig,
        LlamaForCausalLM,
        dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        ),
    )
    yield (
        "gpt_neox",
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
        ),
    )
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM
    except Exception:
        return
    yield (
        "qwen3",
        Qwen3Config,
        Qwen3ForCausalLM,
        dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        ),
    )


def check_tiny_architectures() -> None:
    for name, config_cls, model_cls, kwargs in _tiny_architectures():
        torch.manual_seed(0)
        config = config_cls(**kwargs)
        config._attn_implementation = "eager"
        model = model_cls(config).eval()
        windows = torch.randint(5, 60, (3, 24), dtype=torch.long)
        windows[:, 0] = 1

        selection = identify_sink_heads(model, windows, batch_size=1, query_start=16, topk=2, show_progress=False)
        assert 0 <= selection.sink_position < windows.shape[1]
        assert len(selection.sink_heavy) == 2
        assert len(selection.low_sink) == 2
        assert len(selection.all_scores) == config.num_hidden_layers * config.num_attention_heads

        input_ids = windows[0]
        targets = choose_target_positions(input_ids, 18, sink_position=selection.sink_position, special_ids={1}, rng=random.Random(3))
        assert targets is not None
        head = selection.sink_heavy[0]
        target_token_id = int(input_ids[19])

        attention_result = run_attention_site(
            model,
            input_ids,
            site=InterventionSite(0, 18, head.layer, head.head, "sink_heavy", "sink", targets["sink"], 0.02),
            target_token_id=target_token_id,
        )
        assert attention_result.target_kind == "sink"
        assert not attention_result.skipped

        batch_sites = [
            InterventionSite(0, 18, head.layer, head.head, "sink_heavy", "sink", targets["sink"], 0.02),
            InterventionSite(0, 18, head.layer, head.head, "sink_heavy", "early", targets["early"], 0.02),
        ]
        batch_results = run_attention_sites_batch(
            model,
            [input_ids, input_ids],
            sites=batch_sites,
            target_token_ids=[target_token_id, target_token_id],
        )
        assert [row.target_kind for row in batch_results] == ["sink", "early"]
        assert all(not row.skipped for row in batch_results)

        value_result = run_value_site(
            model,
            input_ids,
            site=InterventionSite(0, 18, head.layer, head.head, "sink_heavy", "early", targets["early"], 0.02),
            sink_position=selection.sink_position,
            control_position=targets["early"],
            intervention_kind="ordinary_value_at_sink",
            target_token_id=target_token_id,
        )
        assert value_result.intervention_kind == "ordinary_value_at_sink"

        query_targets = {
            query_position: choose_target_positions(
                input_ids,
                query_position,
                sink_position=selection.sink_position,
                special_ids={1},
                rng=random.Random(query_position),
            )
            for query_position in range(16, 23)
        }
        query_targets = {query_position: target_map for query_position, target_map in query_targets.items() if target_map is not None}
        secondary = run_secondary_sequence(
            model,
            input_ids,
            sequence_index=0,
            heads=selection.sink_heavy,
            query_targets=query_targets,
            target_kind="sink",
            delta=0.02,
        )
        expected_sites = len(query_targets) * len(selection.sink_heavy)
        assert secondary.query_positions == len(query_targets)
        assert secondary.applied_sites + secondary.skipped_sites == expected_sites
        print(
            f"{name}: sink_position={selection.sink_position} "
            f"single_l2={attention_result.logit_l2:.6g} "
            f"value_l2={value_result.logit_l2:.6g} "
            f"secondary_sites={secondary.applied_sites}/{expected_sites}"
        )


def check_value_sequence_statistics() -> None:
    rows = []
    for seq, base in [(0, 0.0), (1, 0.1)]:
        for layer, control in [(0, "early"), (4, "local"), (8, "random")]:
            common = {
                "sequence_index": str(seq),
                "query_position": "18",
                "layer": str(layer),
                "head": "0",
                "control_kind": control,
                "sink_position": "0",
                "control_position": "3",
                "delta_loss": "0.0",
                "logit_l2": "0.0",
                "delta_correct_prob": "0.0",
                "skipped": "False",
            }
            rows.append({**common, "intervention_kind": "ordinary_value_at_sink", "kl_clean_intervened": str(0.3 + base)})
            rows.append({**common, "intervention_kind": "sink_value_at_ordinary", "kl_clean_intervened": str(0.1 + base)})
    summaries = runner.summarize_value_sequence_pairs(rows)
    assert {row["layer_band"] for row in summaries} >= {"all", "early", "middle", "late"}
    boot = runner.bootstrap_value_sequence_summaries(summaries, seed=0, samples=20)
    assert {row["layer_band"] for row in boot} >= {"all", "early", "middle", "late"}
    assert all(row["mean_ordinary_at_sink_minus_sink_at_ordinary_kl"] > 0 for row in boot)



def check_secondary_sequence_statistics() -> None:
    rows = []
    for seq, base in [(0, 0.0), (1, 0.1)]:
        for target_kind, kl in [("sink", 0.1 + base), ("early", 0.3 + base), ("local", 0.25 + base), ("random", 0.2 + base)]:
            rows.append(
                {
                    "sequence_index": str(seq),
                    "target_kind": target_kind,
                    "delta": "0.05",
                    "mean_kl_clean_intervened": str(kl),
                    "mean_delta_loss": str(kl / 10),
                    "mean_logit_l2": str(kl * 2),
                    "mean_delta_correct_prob": "0.0",
                    "applied_sites": "10",
                    "skipped_sites": "0",
                    "query_positions": "5",
                }
            )
    summaries = runner.summarize_secondary_sequence_pairs(rows)
    assert len(summaries) == 6
    boot = runner.bootstrap_secondary_sequence_summaries(summaries, seed=0, samples=20)
    assert {row["control_kind"] for row in boot} == {"early", "local", "random"}
    assert all(row["mean_control_minus_sink_kl"] > 0 for row in boot)



def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("placeholder\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_output_auditor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model = "tiny_model"
        dataset = "tiny_dataset"
        delta = 0.05
        (out / model / dataset).mkdir(parents=True)
        (out / "causal_noop_experiment.md").write_text("# report\n")
        (out / "run_summary.json").write_text(json.dumps({"rows": [{"status": "ok", "model_key": model, "dataset": dataset}]}) + "\n")
        (out / model / "head_selection.json").write_text(
            json.dumps(
                {
                    "sink_position": 0,
                    "sink_heavy": [{"layer": 0, "head": 0, "sink_strength": 0.9}, {"layer": 1, "head": 0, "sink_strength": 0.8}],
                    "low_sink": [{"layer": 0, "head": 1, "sink_strength": 0.1}, {"layer": 1, "head": 1, "sink_strength": 0.2}],
                    "all_scores": [
                        {"layer": 0, "head": 0, "sink_strength": 0.9},
                        {"layer": 0, "head": 1, "sink_strength": 0.1},
                        {"layer": 1, "head": 0, "sink_strength": 0.8},
                        {"layer": 1, "head": 1, "sink_strength": 0.2},
                    ],
                    "metadata": {
                        "selection_dataset": "wikitext_train",
                        "dataset": "Salesforce/wikitext",
                        "config": "wikitext-103-raw-v1",
                        "split": "train",
                        "query_start": 16,
                        "topk": 2,
                    },
                }
            )
            + "\n"
        )
        dataset_dir = out / model / dataset
        for filename in ("sequence_summaries.csv", "value_sequence_summaries.csv", "secondary_sequence_summaries.csv"):
            _write_csv(dataset_dir / filename, [])

        site_rows = []
        for seq in (0, 1):
            for head_kind in ("sink_heavy", "low_sink"):
                for target_kind, target_position in (("sink", 0), ("early", 1), ("local", 12), ("random", 7)):
                    site_rows.append(
                        {
                            "sequence_index": seq,
                            "query_position": 16,
                            "layer": 0,
                            "head": 0 if head_kind == "sink_heavy" else 1,
                            "head_kind": head_kind,
                            "target_kind": target_kind,
                            "target_position": target_position,
                            "delta": delta,
                            "kl_clean_intervened": 0.1,
                            "delta_loss": 0.01,
                            "logit_l2": 0.2,
                            "delta_correct_prob": 0.0,
                            "skipped": False,
                        }
                    )
        _write_csv(dataset_dir / "site_results.csv", site_rows)

        value_raw_rows = []
        for seq in (0, 1):
            for control_kind, control_position in (("early", 1), ("local", 12), ("random", 7)):
                for intervention_kind in ("ordinary_value_at_sink", "sink_value_at_ordinary"):
                    value_raw_rows.append(
                        {
                            "sequence_index": seq,
                            "query_position": 16,
                            "layer": 0,
                            "head": 0,
                            "control_kind": control_kind,
                            "sink_position": 0,
                            "control_position": control_position,
                            "intervention_kind": intervention_kind,
                            "kl_clean_intervened": 0.1,
                            "delta_loss": 0.01,
                            "logit_l2": 0.2,
                            "delta_correct_prob": 0.0,
                            "skipped": False,
                        }
                    )
        _write_csv(dataset_dir / "value_content_results.csv", value_raw_rows)

        primary_rows = []
        for head_kind in ("sink_heavy", "low_sink"):
            for control_kind in ("early", "local", "random"):
                primary_rows.append(
                    {
                        "delta": delta,
                        "head_kind": head_kind,
                        "layer_band": "all",
                        "control_kind": control_kind,
                        "mean_paired_difference": 0.1,
                        "ci_low": 0.0,
                        "ci_high": 0.2,
                        "mean_control_minus_sink_loss": 0.01,
                        "mean_control_minus_sink_logit_l2": 0.3,
                        "mean_control_minus_sink_correct_prob": -0.02,
                        "fraction_sink_less_disruptive": 1.0,
                        "num_sequences": 2,
                    }
                )
        _write_csv(dataset_dir / "bootstrap_summaries.csv", primary_rows)

        value_rows = []
        for control_kind in ("early", "local", "random"):
            value_rows.append(
                {
                    "control_kind": control_kind,
                    "layer_band": "all",
                    "mean_ordinary_at_sink_minus_sink_at_ordinary_kl": 0.1,
                    "ci_low": 0.0,
                    "ci_high": 0.2,
                    "mean_ordinary_at_sink_minus_sink_at_ordinary_loss": 0.01,
                    "mean_ordinary_at_sink_minus_sink_at_ordinary_logit_l2": 0.3,
                    "fraction_ordinary_at_sink_more_disruptive": 1.0,
                    "num_sequences": 2,
                }
            )
        _write_csv(dataset_dir / "value_bootstrap_summaries.csv", value_rows)

        secondary_rows = []
        for seq in (0, 1):
            for target_kind in ("sink", "early", "local", "random"):
                secondary_rows.append(
                    {
                        "sequence_index": seq,
                        "target_kind": target_kind,
                        "delta": delta,
                        "mean_kl_clean_intervened": 0.1,
                        "mean_delta_loss": 0.01,
                        "mean_logit_l2": 0.2,
                        "mean_delta_correct_prob": 0.0,
                        "applied_sites": 4,
                        "skipped_sites": 0,
                        "query_positions": 2,
                    }
                )
        _write_csv(dataset_dir / "secondary_results.csv", secondary_rows)

        secondary_boot_rows = []
        for control_kind in ("early", "local", "random"):
            secondary_boot_rows.append(
                {
                    "delta": delta,
                    "control_kind": control_kind,
                    "mean_control_minus_sink_kl": 0.1,
                    "ci_low": 0.0,
                    "ci_high": 0.2,
                    "mean_control_minus_sink_loss": 0.01,
                    "mean_control_minus_sink_logit_l2": 0.2,
                    "fraction_sink_less_disruptive": 1.0,
                    "num_sequences": 2,
                }
            )
        _write_csv(dataset_dir / "secondary_bootstrap_summaries.csv", secondary_boot_rows)

        result = auditor.audit_outputs(
            out,
            models=(model,),
            datasets=(dataset,),
            deltas=(delta,),
            expected_heads=2,
            min_sequences=2,
        )
        assert result.ok, result.errors



def check_report_row_merge() -> None:
    existing = [
        {"status": "ok", "model_key": "pythia_1b", "dataset": "wikitext_test", "version": "old"},
        {"status": "ok", "model_key": "llama3_2_3b", "dataset": "pile_val", "version": "keep"},
    ]
    current = [
        {"status": "ok", "model_key": "pythia_1b", "dataset": "wikitext_test", "version": "new"},
        {"status": "ok", "model_key": "qwen3_4b", "dataset": "pile_val", "version": "add"},
    ]
    merged = runner.merge_report_rows(existing, current)
    assert [(row["model_key"], row["dataset"]) for row in merged] == [
        ("pythia_1b", "wikitext_test"),
        ("llama3_2_3b", "pile_val"),
        ("qwen3_4b", "pile_val"),
    ]
    assert merged[0]["version"] == "new"
    assert merged[1]["version"] == "keep"
    assert merged[2]["version"] == "add"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_summary.json"
        path.write_text(json.dumps({"rows": existing}) + "\n")
        loaded = runner.load_existing_report_rows(path)
        assert loaded == existing
        assert runner.load_existing_report_rows(Path(tmp) / "missing.json") == []


def check_report_writer() -> None:
    report_row = {
        "status": "ok",
        "model_key": "pythia_1b",
        "dataset": "pile_val",
        "bootstrap": [
            {
                "head_kind": "sink_heavy",
                "layer_band": "all",
                "control_kind": "early",
                "delta": 0.05,
                "mean_paired_difference": 0.1,
                "ci_low": 0.0,
                "ci_high": 0.2,
                "mean_control_minus_sink_loss": 0.01,
                "mean_control_minus_sink_logit_l2": 0.4,
                "mean_control_minus_sink_correct_prob": -0.03,
                "fraction_sink_less_disruptive": 0.75,
                "num_sequences": 2,
            }
        ],
        "value_summary": [
            {
                "control_kind": "early",
                "intervention_kind": "ordinary_value_at_sink",
                "mean_kl": 0.2,
                "mean_delta_loss": 0.03,
                "mean_logit_l2": 1.2,
                "num_sites": 4,
            }
        ],
        "value_paired_summary": [
            {
                "control_kind": "early",
                "layer_band": "all",
                "mean_ordinary_at_sink_minus_sink_at_ordinary_kl": 0.12,
                "ci_low": 0.02,
                "ci_high": 0.22,
                "fraction_ordinary_at_sink_more_disruptive": 0.8,
                "num_sequences": 5,
            }
        ],
        "secondary_summary": [
            {
                "target_kind": "sink",
                "delta": 0.05,
                "mean_kl": 0.1,
                "mean_delta_loss": 0.01,
                "mean_logit_l2": 1.0,
                "applied_sites": 20,
                "skipped_sites": 1,
                "num_sequences": 2,
            }
        ],
        "secondary_paired_summary": [
            {
                "control_kind": "early",
                "delta": 0.05,
                "mean_control_minus_sink_kl": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
                "fraction_sink_less_disruptive": 1.0,
                "num_sequences": 2,
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.md"
        runner.write_report(path, [report_row])
        text = path.read_text()
        assert "layer band" in text
        assert "mean loss diff" in text
        assert "Value-Content Validation" in text
        assert "Value-Content Paired Asymmetry" in text
        assert "Secondary All-Head Intervention" in text
        assert "Secondary Paired Differences" in text


def main() -> None:
    check_core_math()
    check_fixed_pile_text_windows()
    check_sharded_eval_helpers()
    check_sequence_statistics()
    check_value_sequence_statistics()
    check_secondary_sequence_statistics()
    check_output_auditor()
    check_report_row_merge()
    check_tiny_architectures()
    check_report_writer()
    print("causal noop verifier passed")


if __name__ == "__main__":
    main()
