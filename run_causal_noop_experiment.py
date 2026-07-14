from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/registers_llms/.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("TMPDIR", "/workspace/registers_llms/.tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from sink_neurons.artifacts import load_pt, save_json, save_pt
from sink_neurons.attention_noop import (
    HeadRef,
    HeadSelection,
    InterventionSite,
    bootstrap_sequence_summaries,
    choose_target_positions,
    identify_sink_heads,
    layer_band,
    run_attention_site,
    run_attention_sites_batch,
    run_secondary_sequence,
    run_value_site,
    summarize_sequence_pairs,
)
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import load_model, load_tokenizer
from sink_neurons.windows import build_windows_tensor, resolve_start_token


MODEL_IDS = {
    "pythia_1b": "EleutherAI/pythia-1b",
    "llama3_2_3b": "unsloth/Llama-3.2-3B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "mistral_7b_v0_1": "mistralai/Mistral-7B-v0.1",
}

DATASET_SPLITS = {
    "wikitext_train": "train",
    "wikitext_test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal no-op attention-sink experiment.")
    parser.add_argument("--models", nargs="+", default=["pythia_1b"], choices=sorted(MODEL_IDS))
    parser.add_argument("--out", type=Path, default=Path("outputs/causal_noop_experiment"))
    parser.add_argument("--head-train-windows", type=int, default=256)
    parser.add_argument("--eval-windows", type=int, default=256)
    parser.add_argument("--eval-start", type=int, default=0, help="Global held-out window offset for sharded evaluation runs.")
    parser.add_argument("--sensitivity-windows", type=int, default=128)
    parser.add_argument("--value-windows", type=int, default=128)
    parser.add_argument("--secondary-windows", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--site-batch-size", type=int, default=12, help="Number of one-intervention replicated examples per primary forward batch.")
    parser.add_argument("--query-start", type=int, default=16)
    parser.add_argument("--queries-per-sequence", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--primary-delta", type=float, default=0.05)
    parser.add_argument("--sensitivity-deltas", nargs="+", type=float, default=[0.02, 0.10])
    parser.add_argument("--pilot-deltas", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--run-delta-pilot", action="store_true", help="Run/report the configured pilot deltas; choose frozen deltas from the pilot before full comparisons.")
    parser.add_argument("--deltas", nargs="+", type=float, default=None, help="Override all primary/sensitivity deltas explicitly.")
    parser.add_argument("--datasets", nargs="+", default=["wikitext_test"], choices=["wikitext_test", "pile_val"])
    parser.add_argument("--pile-windows", type=Path, default=None, help="Legacy PT artifact containing pretokenized Pile windows.")
    parser.add_argument("--pile-texts", type=Path, default=None, help="JSON artifact containing the fixed Pile validation text subset.")
    parser.add_argument("--pile-text-examples", type=int, default=2048, help="Non-empty Pile validation rows to cache for the fixed text subset.")
    parser.add_argument("--pile-dataset", default="EleutherAI/pile")
    parser.add_argument("--pile-config", default=None)
    parser.add_argument("--pile-split", default="validation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--force-heads", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-value-validation", action="store_true")
    parser.add_argument("--skip-secondary", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush incremental CSV results every N rows.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _window_cache_path(out: Path, model_key: str, split_name: str, count: int, length: int, stride: int) -> Path:
    return out / model_key / "windows" / f"{split_name}_{count}x{length}_s{stride}.pt"


def load_or_build_wikitext_windows(
    *,
    out: Path,
    model_key: str,
    model_id: str,
    split_name: str,
    count: int,
    length: int,
    stride: int,
    show_progress: bool,
) -> torch.Tensor:
    path = _window_cache_path(out, model_key, split_name, count, length, stride)
    if path.exists():
        payload = load_pt(path)
        windows = payload.get("tensors", {}).get("windows")
        if windows is None:
            windows = payload.get("windows")
        if isinstance(windows, torch.Tensor):
            return windows.long()
    tokenizer = load_tokenizer(model_id)
    windows, token_count = build_windows_tensor(
        tokenizer,
        split=DATASET_SPLITS[split_name],
        window_length=length,
        stride=stride,
        max_windows=count,
        show_progress=show_progress,
    )
    save_pt(
        path,
        {
            "tensors": {"windows": windows},
            "metadata": {
                "model_id": model_id,
                "dataset": "Salesforce/wikitext",
                "config": "wikitext-103-raw-v1",
                "split": DATASET_SPLITS[split_name],
                "tokenized_stream_length": int(token_count),
                "num_windows": int(windows.shape[0]),
                "window_length": int(length),
                "stride": int(stride),
            },
        },
    )
    return windows.long()


def load_or_build_pile_windows(
    *,
    out: Path,
    model_key: str,
    model_id: str,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    count: int,
    length: int,
    stride: int,
    show_progress: bool,
) -> torch.Tensor:
    safe_dataset = dataset_name.replace("/", "_")
    safe_config = "default" if dataset_config is None else dataset_config.replace("/", "_")
    cache_path = out / model_key / "windows" / f"pile_{safe_dataset}_{safe_config}_{split}_{count}x{length}_s{stride}.pt"
    if cache_path.exists():
        payload = load_pt(cache_path)
        windows = payload.get("tensors", {}).get("windows")
        if windows is None:
            windows = payload.get("windows")
        if isinstance(windows, torch.Tensor):
            return windows.long()

    from datasets import load_dataset

    tokenizer = load_tokenizer(model_id)
    start_token = resolve_start_token(tokenizer)
    content_length = int(length) - 1
    if content_length <= 0:
        raise ValueError("window length must be at least 2")
    ds_kwargs = {"split": split}
    if dataset_config is None:
        dataset = load_dataset(dataset_name, **ds_kwargs)
    else:
        dataset = load_dataset(dataset_name, dataset_config, **ds_kwargs)

    windows: list[list[int]] = []
    token_buffer: list[int] = []
    rows_seen = 0
    token_count = 0
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]
    for item in tqdm(dataset, desc="Building Pile windows", disable=not show_progress, dynamic_ncols=True):
        text = item.get("text") or ""
        if not text.strip():
            continue
        rows_seen += 1
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for token_id in ids:
            token_buffer.append(int(token_id))
            token_count += 1
            while len(token_buffer) >= content_length:
                windows.append([start_token.token_id, *token_buffer[:content_length]])
                if len(windows) >= count:
                    break
                drop = min(stride, len(token_buffer))
                token_buffer = token_buffer[drop:]
            if len(windows) >= count:
                break
        if len(windows) >= count:
            break
    if not windows:
        raise ValueError("no Pile windows were produced")
    tensor = torch.tensor(windows[:count], dtype=torch.long)
    save_pt(
        cache_path,
        {
            "tensors": {"windows": tensor},
            "metadata": {
                "model_id": model_id,
                "dataset": dataset_name,
                "config": dataset_config,
                "split": split,
                "rows_seen": rows_seen,
                "tokenized_stream_length": token_count,
                "num_windows": int(tensor.shape[0]),
                "window_length": int(length),
                "stride": int(stride),
                "start_token_id": int(start_token.token_id),
                "start_token_source": start_token.source,
            },
        },
    )
    return tensor.long()



def _safe_name(value: str | None) -> str:
    if value is None:
        return "default"
    return value.replace("/", "_").replace(":", "_")


def _pile_texts_cache_path(
    out: Path,
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    count: int,
) -> Path:
    return out / "pile_subsets" / f"{_safe_name(dataset_name)}_{_safe_name(dataset_config)}_{_safe_name(split)}_{count}_texts.json"


def load_or_build_pile_texts(
    *,
    out: Path,
    path: Path | None,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    count: int,
    show_progress: bool,
) -> tuple[list[str], Path]:
    cache_path = path or _pile_texts_cache_path(out, dataset_name=dataset_name, dataset_config=dataset_config, split=split, count=count)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        texts = payload.get("texts")
        if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
            raise ValueError(f"invalid Pile text subset artifact: {cache_path}")
        if len(texts) < count:
            raise ValueError(f"need {count} Pile texts, found {len(texts)} in {cache_path}")
        return texts[:count], cache_path

    from datasets import load_dataset

    ds_kwargs = {"split": split}
    if dataset_config is None:
        dataset = load_dataset(dataset_name, **ds_kwargs)
    else:
        dataset = load_dataset(dataset_name, dataset_config, **ds_kwargs)

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]

    texts: list[str] = []
    source_indices: list[int] = []
    for source_index, item in enumerate(tqdm(dataset, desc="Caching fixed Pile texts", disable=not show_progress, dynamic_ncols=True)):
        text = item.get("text") or ""
        if not text.strip():
            continue
        texts.append(text)
        source_indices.append(source_index)
        if len(texts) >= count:
            break
    if len(texts) < count:
        raise ValueError(f"only found {len(texts)} non-empty Pile rows; requested {count}")

    digest = hashlib.sha256("\n<|pile-row|>\n".join(texts).encode("utf-8")).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset": dataset_name,
                    "config": dataset_config,
                    "split": split,
                    "num_texts": len(texts),
                    "source_indices": source_indices,
                    "sha256": digest,
                },
                "texts": texts,
            },
            indent=2,
        )
        + "\n"
    )
    return texts, cache_path


def tokenize_pile_texts_to_windows(
    *,
    out: Path,
    model_key: str,
    model_id: str,
    texts: list[str],
    text_artifact: Path,
    count: int,
    length: int,
    stride: int,
    show_progress: bool,
) -> torch.Tensor:
    text_hash = hashlib.sha256("\n<|pile-row|>\n".join(texts).encode("utf-8")).hexdigest()[:16]
    cache_path = out / model_key / "windows" / f"pile_fixed_texts_{text_hash}_{count}x{length}_s{stride}.pt"
    if cache_path.exists():
        payload = load_pt(cache_path)
        windows = payload.get("tensors", {}).get("windows")
        if windows is None:
            windows = payload.get("windows")
        if isinstance(windows, torch.Tensor):
            return windows.long()

    tokenizer = load_tokenizer(model_id)
    start_token = resolve_start_token(tokenizer)
    content_length = int(length) - 1
    if content_length <= 0:
        raise ValueError("window length must be at least 2")

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]

    windows: list[list[int]] = []
    token_buffer: list[int] = []
    token_count = 0
    rows_used = 0
    for text in tqdm(texts, desc=f"Tokenizing fixed Pile texts for {model_key}", disable=not show_progress, dynamic_ncols=True):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        rows_used += 1
        for token_id in ids:
            token_buffer.append(int(token_id))
            token_count += 1
            while len(token_buffer) >= content_length:
                windows.append([start_token.token_id, *token_buffer[:content_length]])
                if len(windows) >= count:
                    break
                drop = min(stride, len(token_buffer))
                token_buffer = token_buffer[drop:]
            if len(windows) >= count:
                break
        if len(windows) >= count:
            break
    if len(windows) < count:
        raise ValueError(
            f"fixed Pile text subset produced {len(windows)} windows for {model_key}; "
            f"increase --pile-text-examples or lower --eval-windows"
        )

    tensor = torch.tensor(windows[:count], dtype=torch.long)
    save_pt(
        cache_path,
        {
            "tensors": {"windows": tensor},
            "metadata": {
                "model_id": model_id,
                "dataset": "fixed_pile_text_subset",
                "text_artifact": str(text_artifact),
                "text_sha256_prefix": text_hash,
                "rows_available": len(texts),
                "rows_used": rows_used,
                "tokenized_stream_length": token_count,
                "num_windows": int(tensor.shape[0]),
                "window_length": int(length),
                "stride": int(stride),
                "start_token_id": int(start_token.token_id),
                "start_token_source": start_token.source,
            },
        },
    )
    return tensor.long()


def load_pile_windows(path: Path, count: int) -> torch.Tensor:
    payload = load_pt(path)
    windows = payload.get("tensors", {}).get("windows")
    if windows is None:
        windows = payload.get("windows")
    if not isinstance(windows, torch.Tensor):
        raise ValueError(f"no windows tensor in {path}")
    if int(windows.shape[0]) < count:
        raise ValueError(f"need {count} Pile windows, found {int(windows.shape[0])}")
    return windows[:count].long().clone()


def selection_to_payload(selection: HeadSelection) -> dict[str, Any]:
    return {
        "sink_position": selection.sink_position,
        "sink_heavy": [asdict(item) for item in selection.sink_heavy],
        "low_sink": [asdict(item) for item in selection.low_sink],
        "all_scores": [asdict(item) for item in selection.all_scores],
    }


def selection_from_payload(payload: dict[str, Any]) -> HeadSelection:
    return HeadSelection(
        sink_position=int(payload["sink_position"]),
        sink_heavy=tuple(HeadRef(**item) for item in payload["sink_heavy"]),
        low_sink=tuple(HeadRef(**item) for item in payload["low_sink"]),
        all_scores=tuple(HeadRef(**item) for item in payload["all_scores"]),
    )


def load_or_identify_heads(
    *,
    model: torch.nn.Module,
    out: Path,
    model_key: str,
    model_id: str,
    windows: torch.Tensor,
    batch_size: int,
    query_start: int,
    topk: int,
    force: bool,
    show_progress: bool,
) -> HeadSelection:
    path = out / model_key / "head_selection.json"
    if path.exists() and not force:
        return selection_from_payload(json.loads(path.read_text()))
    selection = identify_sink_heads(
        model,  # type: ignore[arg-type]
        windows,
        batch_size=batch_size,
        query_start=query_start,
        topk=topk,
        show_progress=show_progress,
    )
    payload = selection_to_payload(selection)
    payload["metadata"] = {
        "model_key": model_key,
        "model_id": model_id,
        "selection_dataset": "wikitext_train",
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "num_windows": int(windows.shape[0]),
        "window_length": int(windows.shape[1]) if windows.ndim == 2 else None,
        "query_start": int(query_start),
        "topk": int(topk),
        "batch_size": int(batch_size),
    }
    save_json(path, payload)
    return selection


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def site_key_from_site(site: InterventionSite) -> tuple[Any, ...]:
    return (
        int(site.sequence_index),
        int(site.query_position),
        int(site.layer),
        int(site.head),
        site.head_kind,
        site.target_kind,
        int(site.target_position),
        float(site.delta),
    )


def site_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["sequence_index"]),
        int(row["query_position"]),
        int(row["layer"]),
        int(row["head"]),
        row["head_kind"],
        row["target_kind"],
        int(row["target_position"]),
        float(row["delta"]),
    )


def site_result_from_row(row: dict[str, Any]):
    from sink_neurons.attention_noop import SiteResult

    return SiteResult(
        sequence_index=int(row["sequence_index"]),
        query_position=int(row["query_position"]),
        layer=int(row["layer"]),
        head=int(row["head"]),
        head_kind=row["head_kind"],  # type: ignore[arg-type]
        target_kind=row["target_kind"],  # type: ignore[arg-type]
        target_position=int(row["target_position"]),
        delta=float(row["delta"]),
        kl_clean_intervened=float(row["kl_clean_intervened"]),
        delta_loss=float(row["delta_loss"]),
        logit_l2=float(row["logit_l2"]),
        delta_correct_prob=float(row["delta_correct_prob"]),
        skipped=_as_bool(row["skipped"]),
    )


def value_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["sequence_index"]),
        int(row["query_position"]),
        int(row["layer"]),
        int(row["head"]),
        row["control_kind"],
        int(row["sink_position"]),
        int(row["control_position"]),
        row["intervention_kind"],
    )


def secondary_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (int(row["sequence_index"]), row["target_kind"], float(row["delta"]))


def sampled_queries(length: int, *, query_start: int, count: int, rng: random.Random) -> list[int]:
    eligible = list(range(query_start, length - 1))
    if len(eligible) <= count:
        return eligible
    return sorted(rng.sample(eligible, count))


def _matched_low_sink_head(selection: HeadSelection, sink_head: HeadRef, rng: random.Random) -> HeadRef:
    same_layer = [head for head in selection.low_sink if head.layer == sink_head.layer]
    if same_layer:
        return rng.choice(same_layer)
    return rng.choice(list(selection.low_sink))


def _unique_deltas(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        fval = float(value)
        if not any(abs(fval - existing) < 1e-12 for existing in out):
            out.append(fval)
    return out


def build_sites(
    *,
    windows: torch.Tensor,
    selection: HeadSelection,
    tokenizer: object,
    deltas: list[float],
    queries_per_sequence: int,
    query_start: int,
    seed: int,
    sequence_offset: int = 0,
    primary_delta: float | None = None,
    sensitivity_windows: int | None = None,
    target_kinds: tuple[str, ...] = ("sink", "early", "random"),
) -> list[InterventionSite]:
    del queries_per_sequence  # The sampled protocol uses one eligible query per sequence.
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []))
    rng = random.Random(seed)
    sites: list[InterventionSite] = []
    if not selection.sink_heavy or not selection.low_sink:
        return sites
    all_deltas = _unique_deltas(deltas)
    primary = float(primary_delta if primary_delta is not None else all_deltas[0])
    sensitivity_limit = len(windows) if sensitivity_windows is None else max(0, int(sensitivity_windows))
    for local_seq_idx, input_ids in enumerate(windows):
        seq_idx = int(sequence_offset) + int(local_seq_idx)
        eligible_queries = sampled_queries(int(input_ids.numel()), query_start=query_start, count=max(1, int(input_ids.numel())), rng=rng)
        rng.shuffle(eligible_queries)
        chosen: tuple[int, dict[str, int]] | None = None
        for query_pos in eligible_queries:
            targets = choose_target_positions(
                input_ids,
                query_pos,
                sink_position=selection.sink_position,
                special_ids=special_ids,
                rng=rng,
            )
            if targets is not None and all(kind in targets for kind in target_kinds):
                chosen = (query_pos, targets)
                break
        if chosen is None:
            continue
        query_pos, targets = chosen
        sink_head = rng.choice(list(selection.sink_heavy))
        low_head = _matched_low_sink_head(selection, sink_head, rng)
        head_groups = (("sink_heavy", sink_head), ("low_sink", low_head))
        for delta in all_deltas:
            if abs(float(delta) - primary) > 1e-12 and local_seq_idx >= sensitivity_limit:
                continue
            for head_kind, head in head_groups:
                for target_kind in target_kinds:
                    sites.append(
                        InterventionSite(
                            sequence_index=seq_idx,
                            query_position=query_pos,
                            layer=head.layer,
                            head=head.head,
                            head_kind=head_kind,  # type: ignore[arg-type]
                            target_kind=target_kind,  # type: ignore[arg-type]
                            target_position=int(targets[target_kind]),
                            delta=float(delta),
                        )
                    )
    return sites


def run_dataset(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    model_key: str,
    dataset_name: str,
    windows: torch.Tensor,
    selection: HeadSelection,
    args: argparse.Namespace,
    sequence_offset: int = 0,
) -> dict[str, Path]:
    out_dir = args.out / model_key / dataset_name
    site_csv = out_dir / "site_results.csv"
    summary_csv = out_dir / "sequence_summaries.csv"
    bootstrap_csv = out_dir / "bootstrap_summaries.csv"
    value_csv = out_dir / "value_content_results.csv"
    value_summary_csv = out_dir / "value_sequence_summaries.csv"
    value_bootstrap_csv = out_dir / "value_bootstrap_summaries.csv"
    secondary_csv = out_dir / "secondary_results.csv"
    secondary_summary_csv = out_dir / "secondary_sequence_summaries.csv"
    secondary_bootstrap_csv = out_dir / "secondary_bootstrap_summaries.csv"
    delta_pilot_csv = out_dir / "delta_pilot_summary.csv"
    output_paths = {
        "site_csv": site_csv,
        "summary_csv": summary_csv,
        "bootstrap_csv": bootstrap_csv,
        "value_csv": value_csv,
        "value_summary_csv": value_summary_csv,
        "value_bootstrap_csv": value_bootstrap_csv,
        "secondary_csv": secondary_csv,
        "secondary_summary_csv": secondary_summary_csv,
        "secondary_bootstrap_csv": secondary_bootstrap_csv,
        "delta_pilot_csv": delta_pilot_csv,
    }
    if args.force:
        for csv_path in (
            site_csv,
            summary_csv,
            bootstrap_csv,
            value_csv,
            value_summary_csv,
            value_bootstrap_csv,
            secondary_csv,
            secondary_summary_csv,
            secondary_bootstrap_csv,
            delta_pilot_csv,
        ):
            if csv_path.exists():
                csv_path.unlink()

    sites = build_sites(
        windows=windows,
        selection=selection,
        tokenizer=tokenizer,
        deltas=list(args.deltas),
        queries_per_sequence=int(args.queries_per_sequence),
        query_start=int(args.query_start),
        seed=int(args.seed),
        sequence_offset=int(sequence_offset),
        primary_delta=float(args.primary_delta),
        sensitivity_windows=int(args.sensitivity_windows),
    )
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]
    completed_site_keys = {site_key_from_row(row) for row in read_csv_rows(site_csv)}
    site_buffer: list[dict[str, Any]] = []
    site_fieldnames = list(asdict(sites[0]).keys()) + ["kl_clean_intervened", "delta_loss", "logit_l2", "delta_correct_prob", "skipped"] if sites else []
    pending_batch: list[InterventionSite] = []
    batch_size = max(1, int(getattr(args, "site_batch_size", 1)))
    for site in tqdm(sites, desc=f"{model_key} {dataset_name}", disable=args.no_progress, dynamic_ncols=True):
        key = site_key_from_site(site)
        if key in completed_site_keys:
            continue
        pending_batch.append(site)
        if len(pending_batch) < batch_size:
            continue
        input_batch = [windows[item.sequence_index - int(sequence_offset)] for item in pending_batch]
        target_ids = [int(input_batch[idx][pending_batch[idx].query_position + 1].item()) for idx in range(len(pending_batch))]
        rows = [asdict(row) for row in run_attention_sites_batch(model, input_batch, sites=pending_batch, target_token_ids=target_ids)]  # type: ignore[arg-type]
        site_buffer.extend(rows)
        completed_site_keys.update(site_key_from_site(item) for item in pending_batch)
        pending_batch.clear()
        if len(site_buffer) >= max(1, int(args.flush_every)):
            append_csv_rows(site_csv, site_buffer, site_fieldnames)
            site_buffer.clear()
    if pending_batch:
        input_batch = [windows[item.sequence_index - int(sequence_offset)] for item in pending_batch]
        target_ids = [int(input_batch[idx][pending_batch[idx].query_position + 1].item()) for idx in range(len(pending_batch))]
        rows = [asdict(row) for row in run_attention_sites_batch(model, input_batch, sites=pending_batch, target_token_ids=target_ids)]  # type: ignore[arg-type]
        site_buffer.extend(rows)
        completed_site_keys.update(site_key_from_site(item) for item in pending_batch)
    append_csv_rows(site_csv, site_buffer, site_fieldnames)

    if not args.skip_value_validation:
        completed_value_keys = {value_key_from_row(row) for row in read_csv_rows(value_csv)}
        value_buffer: list[dict[str, Any]] = []
        value_fieldnames = [
            "sequence_index",
            "query_position",
            "layer",
            "head",
            "control_kind",
            "sink_position",
            "control_position",
            "intervention_kind",
            "kl_clean_intervened",
            "delta_loss",
            "logit_l2",
            "delta_correct_prob",
            "skipped",
        ]
        seen_value_sites: set[tuple[int, int, int, int, str, int]] = set()
        for site in tqdm(sites, desc=f"{model_key} {dataset_name} value", disable=args.no_progress, dynamic_ncols=True):
            if site.sequence_index - int(sequence_offset) >= int(args.value_windows):
                continue
            if site.head_kind != "sink_heavy" or site.target_kind != "random":
                continue
            if abs(float(site.delta) - float(args.primary_delta)) > 1e-12:
                continue
            key = (site.sequence_index, site.query_position, site.layer, site.head, site.target_kind, site.target_position)
            if key in seen_value_sites:
                continue
            seen_value_sites.add(key)
            input_ids = windows[site.sequence_index - int(sequence_offset)]
            target_token_id = int(input_ids[site.query_position + 1].item())
            for intervention_kind in ("ordinary_value_at_sink", "sink_value_at_ordinary"):
                value_key = (
                    int(site.sequence_index),
                    int(site.query_position),
                    int(site.layer),
                    int(site.head),
                    site.target_kind,
                    int(selection.sink_position),
                    int(site.target_position),
                    intervention_kind,
                )
                if value_key in completed_value_keys:
                    continue
                value_row = run_value_site(
                    model,  # type: ignore[arg-type]
                    input_ids,
                    site=site,
                    sink_position=selection.sink_position,
                    control_position=site.target_position,
                    intervention_kind=intervention_kind,  # type: ignore[arg-type]
                    target_token_id=target_token_id,
                )
                row = asdict(value_row)
                value_buffer.append(row)
                completed_value_keys.add(value_key)
                if len(value_buffer) >= max(1, int(args.flush_every)):
                    append_csv_rows(value_csv, value_buffer, value_fieldnames)
                    value_buffer.clear()
        append_csv_rows(value_csv, value_buffer, value_fieldnames)

    if not args.skip_value_validation:
        value_sequence_summaries = summarize_value_sequence_pairs(read_csv_rows(value_csv))
        write_csv(value_summary_csv, value_sequence_summaries)
        value_bootstrap = bootstrap_value_sequence_summaries(
            value_sequence_summaries,
            seed=int(args.seed) + 2,
            samples=int(args.bootstrap_samples),
        )
        write_csv(value_bootstrap_csv, value_bootstrap)

    if not args.skip_secondary:
        special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []))
        secondary_rng = random.Random(int(args.seed) + 99_000)
        completed_secondary_keys = {secondary_key_from_row(row) for row in read_csv_rows(secondary_csv)}
        secondary_buffer: list[dict[str, Any]] = []
        secondary_fieldnames = [
            "sequence_index",
            "target_kind",
            "delta",
            "mean_kl_clean_intervened",
            "mean_delta_loss",
            "mean_logit_l2",
            "mean_delta_correct_prob",
            "applied_sites",
            "skipped_sites",
            "query_positions",
        ]
        for local_seq_idx, input_ids in enumerate(windows):
            if local_seq_idx >= int(args.secondary_windows):
                continue
            seq_idx = int(sequence_offset) + int(local_seq_idx)
            query_targets = {}
            for query_pos in range(int(args.query_start), int(input_ids.numel()) - 1):
                targets = choose_target_positions(
                    input_ids,
                    query_pos,
                    sink_position=selection.sink_position,
                    special_ids=special_ids,
                    rng=secondary_rng,
                )
                if targets is not None:
                    query_targets[query_pos] = targets
            if not query_targets:
                continue
            for delta in [float(args.primary_delta)]:
                for target_kind in ("sink", "early", "random"):
                    key = (int(seq_idx), target_kind, float(delta))
                    if key in completed_secondary_keys:
                        continue
                    secondary_row = run_secondary_sequence(
                        model,  # type: ignore[arg-type]
                        input_ids,
                        sequence_index=seq_idx,
                        heads=selection.sink_heavy,
                        query_targets=query_targets,
                        target_kind=target_kind,  # type: ignore[arg-type]
                        delta=float(delta),
                    )
                    row = asdict(secondary_row)
                    secondary_buffer.append(row)
                    completed_secondary_keys.add(key)
                    if len(secondary_buffer) >= max(1, int(args.flush_every)):
                        append_csv_rows(secondary_csv, secondary_buffer, secondary_fieldnames)
                        secondary_buffer.clear()
        append_csv_rows(secondary_csv, secondary_buffer, secondary_fieldnames)
        secondary_sequence_summaries = summarize_secondary_sequence_pairs(read_csv_rows(secondary_csv))
        write_csv(secondary_summary_csv, secondary_sequence_summaries)
        secondary_bootstrap = bootstrap_secondary_sequence_summaries(
            secondary_sequence_summaries,
            seed=int(args.seed) + 3,
            samples=int(args.bootstrap_samples),
        )
        write_csv(secondary_bootstrap_csv, secondary_bootstrap)

    site_rows = read_csv_rows(site_csv)
    sequence_summaries = summarize_sequence_pairs(site_result_from_row(row) for row in site_rows)
    write_csv(summary_csv, [asdict(row) for row in sequence_summaries])
    bootstrap = bootstrap_sequence_summaries(
        sequence_summaries,
        seed=int(args.seed) + 1,
        samples=int(args.bootstrap_samples),
    )
    write_csv(bootstrap_csv, [asdict(row) for row in bootstrap])
    if getattr(args, "run_delta_pilot", False):
        write_csv(delta_pilot_csv, summarize_delta_pilot_csv(site_csv))
    return output_paths


def load_existing_report_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def merge_report_rows(existing: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}
    order: list[tuple[Any, Any]] = []
    anonymous: list[dict[str, Any]] = []
    for row in [*existing, *current]:
        key = (row.get("model_key"), row.get("dataset"))
        if key[0] is None or key[1] is None:
            anonymous.append(row)
            continue
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order] + anonymous


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Causal No-Op Experiment",
        "",
        "Positive paired differences mean the control target was more disruptive than the sink target.",
        "",
        "| model | dataset | head set | layer band | control | delta | mean control-sink KL | 95% CI | mean loss diff | mean logit L2 diff | mean correct-prob diff | sink less disruptive | sequences |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        for summary in row.get("bootstrap", []):
            lines.append(
                "| {model_key} | {dataset} | {head_kind} | {layer_band} | {control_kind} | {delta:.3g} | "
                "{mean_paired_difference:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | "
                "{mean_control_minus_sink_loss:.6g} | {mean_control_minus_sink_logit_l2:.6g} | "
                "{mean_control_minus_sink_correct_prob:.6g} | {fraction_sink_less_disruptive:.3f} | {num_sequences} |".format(
                    model_key=row["model_key"],
                    dataset=row["dataset"],
                    **summary,
                )
            )
    value_rows = [row for row in rows if row.get("status") == "ok" and row.get("value_summary")]
    if value_rows:
        lines += [
            "",
            "## Value-Content Validation",
            "",
            "| model | dataset | control | value intervention | mean KL | mean loss change | mean logit L2 | sites |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for row in value_rows:
            for summary in row.get("value_summary", []):
                lines.append(
                    "| {model_key} | {dataset} | {control_kind} | {intervention_kind} | "
                    "{mean_kl:.6g} | {mean_delta_loss:.6g} | {mean_logit_l2:.6g} | {num_sites} |".format(
                        model_key=row["model_key"],
                        dataset=row["dataset"],
                        **summary,
                    )
                )

    value_paired_rows = [row for row in rows if row.get("status") == "ok" and row.get("value_paired_summary")]
    if value_paired_rows:
        lines += [
            "",
            "## Value-Content Paired Asymmetry",
            "",
            "Positive paired differences mean ordinary value at the sink was more disruptive than sink value at the ordinary position.",
            "",
            "| model | dataset | control | layer band | mean paired KL diff | 95% CI | ordinary-at-sink more disruptive | sequences |",
            "|---|---|---|---|---:|---|---:|---:|",
        ]
        for row in value_paired_rows:
            for summary in row.get("value_paired_summary", []):
                lines.append(
                    "| {model_key} | {dataset} | {control_kind} | {layer_band} | "
                    "{mean_ordinary_at_sink_minus_sink_at_ordinary_kl:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | "
                    "{fraction_ordinary_at_sink_more_disruptive:.3f} | {num_sequences} |".format(
                        model_key=row["model_key"],
                        dataset=row["dataset"],
                        **summary,
                    )
                )

    secondary_rows = [row for row in rows if row.get("status") == "ok" and row.get("secondary_summary")]
    if secondary_rows:
        lines += [
            "",
            "## Secondary All-Head Intervention",
            "",
            "| model | dataset | target | delta | mean KL | mean loss change | mean logit L2 | applied sites | skipped sites | sequences |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in secondary_rows:
            for summary in row.get("secondary_summary", []):
                lines.append(
                    "| {model_key} | {dataset} | {target_kind} | {delta:.3g} | "
                    "{mean_kl:.6g} | {mean_delta_loss:.6g} | {mean_logit_l2:.6g} | "
                    "{applied_sites} | {skipped_sites} | {num_sequences} |".format(
                        model_key=row["model_key"],
                        dataset=row["dataset"],
                        **summary,
                    )
                )

    secondary_paired_rows = [row for row in rows if row.get("status") == "ok" and row.get("secondary_paired_summary")]
    if secondary_paired_rows:
        lines += [
            "",
            "## Secondary Paired Differences",
            "",
            "Positive paired differences mean the all-head ordinary target was more disruptive than the all-head sink target.",
            "",
            "| model | dataset | control | delta | mean control-sink KL | 95% CI | sink less disruptive | sequences |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
        for row in secondary_paired_rows:
            for summary in row.get("secondary_paired_summary", []):
                lines.append(
                    "| {model_key} | {dataset} | {control_kind} | {delta:.3g} | "
                    "{mean_control_minus_sink_kl:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | "
                    "{fraction_sink_less_disruptive:.3f} | {num_sequences} |".format(
                        model_key=row["model_key"],
                        dataset=row["dataset"],
                        **summary,
                    )
                )

    pilot_rows = [row for row in rows if row.get("status") == "ok" and row.get("delta_pilot_summary")]
    if pilot_rows:
        lines += [
            "",
            "## Delta Pilot",
            "",
            "Use this table to choose one measurable, non-pathological primary delta and freeze one smaller/larger sensitivity value across models.",
            "",
            "| model | dataset | delta | head set | target | valid sites | skipped sites | mean KL | std KL | mean loss change | std loss change |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in pilot_rows:
            for summary in row.get("delta_pilot_summary", []):
                lines.append(
                    "| {model_key} | {dataset} | {delta:.3g} | {head_kind} | {target_kind} | "
                    "{valid_sites} | {skipped_sites} | {mean_kl:.6g} | {std_kl:.6g} | "
                    "{mean_delta_loss:.6g} | {std_delta_loss:.6g} |".format(
                        model_key=row["model_key"],
                        dataset=row["dataset"],
                        **summary,
                    )
                )

    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        for row in errors:
            lines.append(f"- `{row.get('model_key')}` `{row.get('dataset')}`: {row.get('error')}")
    path.write_text("\n".join(lines) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))



def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, var ** 0.5


def summarize_delta_pilot_csv(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((float(row["delta"]), row["head_kind"], row["target_kind"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (delta, head_kind, target_kind), group in sorted(grouped.items()):
        valid = [row for row in group if not _as_bool(row.get("skipped", False))]
        kls = [float(row["kl_clean_intervened"]) for row in valid]
        losses = [float(row["delta_loss"]) for row in valid]
        logit_l2s = [float(row["logit_l2"]) for row in valid]
        mean_kl, std_kl = _mean_std(kls)
        mean_loss, std_loss = _mean_std(losses)
        mean_l2, std_l2 = _mean_std(logit_l2s)
        out.append(
            {
                "delta": delta,
                "head_kind": head_kind,
                "target_kind": target_kind,
                "valid_sites": len(valid),
                "skipped_sites": len(group) - len(valid),
                "num_sequences": len({int(row["sequence_index"]) for row in valid}),
                "mean_kl": mean_kl,
                "std_kl": std_kl,
                "mean_delta_loss": mean_loss,
                "std_delta_loss": std_loss,
                "mean_logit_l2": mean_l2,
                "std_logit_l2": std_l2,
            }
        )
    return out


def summarize_value_csv(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["control_kind"], row["intervention_kind"]), []).append(row)
    out = []
    for (control_kind, intervention_kind), group in sorted(grouped.items()):
        if not group:
            continue
        out.append(
            {
                "control_kind": control_kind,
                "intervention_kind": intervention_kind,
                "mean_kl": sum(float(row["kl_clean_intervened"]) for row in group) / len(group),
                "mean_delta_loss": sum(float(row["delta_loss"]) for row in group) / len(group),
                "mean_logit_l2": sum(float(row["logit_l2"]) for row in group) / len(group),
                "num_sites": len(group),
            }
        )
    return out


def summarize_value_sequence_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = [row for row in rows if not _as_bool(row.get("skipped", False))]
    if not parsed:
        return []
    num_layers = max(int(row["layer"]) for row in parsed) + 1
    by_site: dict[tuple[int, int, int, int, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in parsed:
        key = (
            int(row["sequence_index"]),
            int(row["query_position"]),
            int(row["layer"]),
            int(row["head"]),
            row["control_kind"],
            int(row["sink_position"]),
            int(row["control_position"]),
        )
        by_site.setdefault(key, {})[row["intervention_kind"]] = row

    by_sequence: dict[tuple[int, str, str], list[tuple[float, float, float]]] = {}
    for (seq, _query, layer, _head, control_kind, _sink_pos, _control_pos), interventions in by_site.items():
        ordinary_at_sink = interventions.get("ordinary_value_at_sink")
        sink_at_ordinary = interventions.get("sink_value_at_ordinary")
        if ordinary_at_sink is None or sink_at_ordinary is None:
            continue
        diff_kl = float(ordinary_at_sink["kl_clean_intervened"]) - float(sink_at_ordinary["kl_clean_intervened"])
        diff_loss = float(ordinary_at_sink["delta_loss"]) - float(sink_at_ordinary["delta_loss"])
        diff_logit_l2 = float(ordinary_at_sink["logit_l2"]) - float(sink_at_ordinary["logit_l2"])
        for band in ("all", layer_band(layer, num_layers)):
            by_sequence.setdefault((seq, control_kind, band), []).append((diff_kl, diff_loss, diff_logit_l2))

    out: list[dict[str, Any]] = []
    for (seq, control_kind, band), diffs in sorted(by_sequence.items()):
        n = len(diffs)
        out.append(
            {
                "sequence_index": seq,
                "control_kind": control_kind,
                "layer_band": band,
                "mean_ordinary_at_sink_minus_sink_at_ordinary_kl": sum(item[0] for item in diffs) / n,
                "mean_ordinary_at_sink_minus_sink_at_ordinary_loss": sum(item[1] for item in diffs) / n,
                "mean_ordinary_at_sink_minus_sink_at_ordinary_logit_l2": sum(item[2] for item in diffs) / n,
                "ordinary_at_sink_more_disruptive": (sum(item[0] for item in diffs) / n) > 0.0,
                "num_pairs": n,
            }
        )
    return out


def bootstrap_value_sequence_summaries(
    summaries: list[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault((row["control_kind"], row["layer_band"]), []).append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for (control_kind, band), rows in sorted(grouped.items()):
        vals = [float(row["mean_ordinary_at_sink_minus_sink_at_ordinary_kl"]) for row in rows]
        losses = [float(row["mean_ordinary_at_sink_minus_sink_at_ordinary_loss"]) for row in rows]
        logit_l2s = [float(row["mean_ordinary_at_sink_minus_sink_at_ordinary_logit_l2"]) for row in rows]
        n = len(vals)
        if n == 0:
            continue
        means = []
        for _ in range(samples):
            draw = [vals[rng.randrange(n)] for _ in range(n)]
            means.append(sum(draw) / n)
        means.sort()
        lo = means[max(0, int(0.025 * samples) - 1)]
        hi = means[min(samples - 1, int(0.975 * samples))]
        out.append(
            {
                "control_kind": control_kind,
                "layer_band": band,
                "mean_ordinary_at_sink_minus_sink_at_ordinary_kl": sum(vals) / n,
                "ci_low": lo,
                "ci_high": hi,
                "mean_ordinary_at_sink_minus_sink_at_ordinary_loss": sum(losses) / n,
                "mean_ordinary_at_sink_minus_sink_at_ordinary_logit_l2": sum(logit_l2s) / n,
                "fraction_ordinary_at_sink_more_disruptive": sum(value > 0.0 for value in vals) / n,
                "num_sequences": n,
            }
        )
    return out


def summarize_secondary_sequence_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sequence_delta: dict[tuple[int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_sequence_delta.setdefault((int(row["sequence_index"]), float(row["delta"])), {})[row["target_kind"]] = row
    out: list[dict[str, Any]] = []
    for (sequence_index, delta), target_rows in sorted(by_sequence_delta.items()):
        sink = target_rows.get("sink")
        if sink is None:
            continue
        sink_kl = float(sink["mean_kl_clean_intervened"])
        sink_loss = float(sink["mean_delta_loss"])
        sink_logit_l2 = float(sink["mean_logit_l2"])
        for control_kind in ("early", "local", "random"):
            control = target_rows.get(control_kind)
            if control is None:
                continue
            diff_kl = float(control["mean_kl_clean_intervened"]) - sink_kl
            out.append(
                {
                    "sequence_index": sequence_index,
                    "delta": delta,
                    "control_kind": control_kind,
                    "mean_control_minus_sink_kl": diff_kl,
                    "mean_control_minus_sink_loss": float(control["mean_delta_loss"]) - sink_loss,
                    "mean_control_minus_sink_logit_l2": float(control["mean_logit_l2"]) - sink_logit_l2,
                    "sink_less_disruptive": diff_kl > 0.0,
                    "sink_applied_sites": int(sink["applied_sites"]),
                    "control_applied_sites": int(control["applied_sites"]),
                }
            )
    return out


def bootstrap_secondary_sequence_summaries(
    summaries: list[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault((float(row["delta"]), row["control_kind"]), []).append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for (delta, control_kind), rows in sorted(grouped.items()):
        vals = [float(row["mean_control_minus_sink_kl"]) for row in rows]
        losses = [float(row["mean_control_minus_sink_loss"]) for row in rows]
        logit_l2s = [float(row["mean_control_minus_sink_logit_l2"]) for row in rows]
        n = len(vals)
        if n == 0:
            continue
        means = []
        for _ in range(samples):
            draw = [vals[rng.randrange(n)] for _ in range(n)]
            means.append(sum(draw) / n)
        means.sort()
        lo = means[max(0, int(0.025 * samples) - 1)]
        hi = means[min(samples - 1, int(0.975 * samples))]
        out.append(
            {
                "delta": delta,
                "control_kind": control_kind,
                "mean_control_minus_sink_kl": sum(vals) / n,
                "ci_low": lo,
                "ci_high": hi,
                "mean_control_minus_sink_loss": sum(losses) / n,
                "mean_control_minus_sink_logit_l2": sum(logit_l2s) / n,
                "fraction_sink_less_disruptive": sum(value > 0.0 for value in vals) / n,
                "num_sequences": n,
            }
        )
    return out


def summarize_secondary_csv(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["target_kind"], float(row["delta"])), []).append(row)
    out = []
    for (target_kind, delta), group in sorted(grouped.items()):
        if not group:
            continue
        out.append(
            {
                "target_kind": target_kind,
                "delta": float(delta),
                "mean_kl": sum(float(row["mean_kl_clean_intervened"]) for row in group) / len(group),
                "mean_delta_loss": sum(float(row["mean_delta_loss"]) for row in group) / len(group),
                "mean_logit_l2": sum(float(row["mean_logit_l2"]) for row in group) / len(group),
                "applied_sites": sum(int(row["applied_sites"]) for row in group),
                "skipped_sites": sum(int(row["skipped_sites"]) for row in group),
                "num_sequences": len({int(row["sequence_index"]) for row in group}),
            }
        )
    return out


def slice_eval_windows(windows: torch.Tensor, *, start: int, count: int) -> torch.Tensor:
    if start < 0:
        raise ValueError("--eval-start must be nonnegative")
    if count <= 0:
        raise ValueError("--eval-windows must be positive")
    end = start + count
    if int(windows.shape[0]) < end:
        raise ValueError(f"need {end} evaluation windows for offset slice, found {int(windows.shape[0])}")
    return windows[start:end].long().clone()



def main() -> None:
    configure_runtime()
    args = parse_args()
    if args.run_delta_pilot:
        args.deltas = _unique_deltas([float(delta) for delta in args.pilot_deltas])
        args.sensitivity_windows = int(args.eval_windows)
        args.skip_value_validation = True
        args.skip_secondary = True
    elif args.deltas is None:
        args.deltas = _unique_deltas([float(args.primary_delta), *[float(delta) for delta in args.sensitivity_deltas]])
    else:
        args.deltas = _unique_deltas([float(delta) for delta in args.deltas])
    eval_start = int(args.eval_start)
    eval_count = int(args.eval_windows)
    eval_total = eval_start + eval_count
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "run_summary.json"
    report_path = args.out / "causal_noop_experiment.md"
    existing_report_rows = load_existing_report_rows(summary_path)
    report_rows: list[dict[str, Any]] = []
    for model_key in args.models:
        model_id = MODEL_IDS[model_key]
        print(f"[load] {model_key} {model_id}", flush=True)
        model = load_model(model_id, eager_attention=True)
        tokenizer = load_tokenizer(model_id)
        try:
            head_windows = load_or_build_wikitext_windows(
                out=args.out,
                model_key=model_key,
                model_id=model_id,
                split_name="wikitext_train",
                count=int(args.head_train_windows),
                length=int(args.window_length),
                stride=int(args.stride),
                show_progress=not args.no_progress,
            )
            selection = load_or_identify_heads(
                model=model,
                out=args.out,
                model_key=model_key,
                model_id=model_id,
                windows=head_windows,
                batch_size=int(args.batch_size),
                query_start=int(args.query_start),
                topk=int(args.heads),
                force=bool(args.force_heads),
                show_progress=not args.no_progress,
            )
            for dataset_name in args.datasets:
                try:
                    if dataset_name == "wikitext_test":
                        eval_windows = load_or_build_wikitext_windows(
                            out=args.out,
                            model_key=model_key,
                            model_id=model_id,
                            split_name="wikitext_test",
                            count=eval_total,
                            length=int(args.window_length),
                            stride=int(args.stride),
                            show_progress=not args.no_progress,
                        )
                    else:
                        if args.pile_windows is not None:
                            eval_windows = load_pile_windows(args.pile_windows, eval_total)
                        else:
                            pile_texts, pile_text_artifact = load_or_build_pile_texts(
                                out=args.out,
                                path=args.pile_texts,
                                dataset_name=str(args.pile_dataset),
                                dataset_config=args.pile_config,
                                split=str(args.pile_split),
                                count=int(args.pile_text_examples),
                                show_progress=not args.no_progress,
                            )
                            eval_windows = tokenize_pile_texts_to_windows(
                                out=args.out,
                                model_key=model_key,
                                model_id=model_id,
                                texts=pile_texts,
                                text_artifact=pile_text_artifact,
                                count=eval_total,
                                length=int(args.window_length),
                                stride=int(args.stride),
                                show_progress=not args.no_progress,
                            )
                    eval_windows = slice_eval_windows(eval_windows, start=eval_start, count=eval_count)
                    paths = run_dataset(
                        model=model,
                        tokenizer=tokenizer,
                        model_key=model_key,
                        dataset_name=dataset_name,
                        windows=eval_windows,
                        selection=selection,
                        args=args,
                        sequence_offset=eval_start,
                    )
                    bootstrap = [
                        {
                            key: (float(value) if key not in {"head_kind", "layer_band", "control_kind"} else value)
                            for key, value in row.items()
                        }
                        for row in read_csv_rows(paths["bootstrap_csv"])
                    ]
                    value_summary = [] if args.skip_value_validation else summarize_value_csv(paths["value_csv"])
                    value_paired_summary = [] if args.skip_value_validation else [
                        {
                            key: (float(value) if key not in {"control_kind", "layer_band"} else value)
                            for key, value in row.items()
                        }
                        for row in read_csv_rows(paths["value_bootstrap_csv"])
                    ]
                    secondary_summary = [] if args.skip_secondary else summarize_secondary_csv(paths["secondary_csv"])
                    secondary_paired_summary = [] if args.skip_secondary else [
                        {
                            key: (float(value) if key not in {"control_kind"} else value)
                            for key, value in row.items()
                        }
                        for row in read_csv_rows(paths["secondary_bootstrap_csv"])
                    ]
                    delta_pilot_summary = [
                        {
                            key: (float(value) if key not in {"head_kind", "target_kind"} else value)
                            for key, value in row.items()
                        }
                        for row in read_csv_rows(paths["delta_pilot_csv"])
                    ]
                    report_rows.append(
                        {
                            "status": "ok",
                            "model_key": model_key,
                            "model_id": model_id,
                            "dataset": dataset_name,
                            "sink_position": selection.sink_position,
                            "bootstrap": bootstrap,
                            "value_summary": value_summary,
                            "value_paired_summary": value_paired_summary,
                            "secondary_summary": secondary_summary,
                            "secondary_paired_summary": secondary_paired_summary,
                            "delta_pilot_summary": delta_pilot_summary,
                        }
                    )
                except Exception as exc:
                    report_rows.append(
                        {
                            "status": "error",
                            "model_key": model_key,
                            "dataset": dataset_name,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[error] {model_key} {dataset_name}: {exc!r}", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    merged_report_rows = merge_report_rows(existing_report_rows, report_rows)
    save_json(summary_path, {"rows": merged_report_rows})
    write_report(report_path, merged_report_rows)
    print(report_path)


if __name__ == "__main__":
    main()
