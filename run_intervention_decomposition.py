from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/registers_llms/.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

print("[startup] importing torch", flush=True)
import torch
import torch.nn.functional as F

print("[startup] importing sink_neurons.bias_transfer", flush=True)
import sink_neurons.bias_transfer as bt


MODEL_KEYS = ("pythia_1b", "llama3_2_3b", "qwen3_4b")
CONDITIONS = (
    ("dummy_only", "selected", None),
    ("source_subtract_only", "selected", "subtract_only"),
    ("dummy_add_only", "selected", "add_only"),
    ("full_transfer", "selected", "transfer"),
    ("random_dummy_add_only", "layer_matched_random", "add_only"),
    ("random_full_transfer", "layer_matched_random", "transfer"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    preferred = [
        "model_key", "condition", "candidate_group", "example_idx", "mode", "scale",
        "clean_nll", "intervened_nll", "delta_nll", "output_kl", "bos_attention",
        "dummy_attention", "dummy_minus_bos", "tokens",
    ]
    all_keys = sorted({key for row in rows for key in row})
    keys.extend([key for key in preferred if key in all_keys])
    keys.extend([key for key in all_keys if key not in keys])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_first_row(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one accepted test row in {path}, found {len(rows)}")
    row = rows[0]
    if row.get("status") != "ok" or row.get("mode") != "transfer":
        raise ValueError(f"unexpected accepted row in {path}: status={row.get('status')} mode={row.get('mode')}")
    return row


def cfg_for_key(model_key: str) -> Any:
    for cfg in bt.CONFIGS:
        if cfg.key == model_key:
            return cfg
    raise KeyError(model_key)


def layer_counts(selected: torch.Tensor) -> str:
    counts: dict[int, int] = {}
    for layer in selected[:, 0].tolist():
        counts[int(layer)] = counts.get(int(layer), 0) + 1
    return " ".join(f"{layer}:{counts[layer]}" for layer in sorted(counts))


def sample_layer_matched_random(score_rows: list[Any], selected: torch.Tensor, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    selected_pairs = {(int(layer), int(neuron)) for layer, neuron in selected.tolist()}
    by_layer: dict[int, list[int]] = {}
    for row in score_rows:
        layer = int(row[1])
        neuron = int(row[2])
        if (layer, neuron) in selected_pairs:
            continue
        by_layer.setdefault(layer, []).append(neuron)
    sampled: list[tuple[int, int]] = []
    for layer in sorted({int(x) for x in selected[:, 0].tolist()}):
        need = int((selected[:, 0] == layer).sum().item())
        pool = sorted(set(by_layer.get(layer, [])))
        if len(pool) < need:
            raise ValueError(f"not enough random candidates for layer {layer}: need {need}, have {len(pool)}")
        sampled.extend((layer, neuron) for neuron in rng.sample(pool, need))
    return torch.tensor(sampled, dtype=torch.long)


def per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return (losses * valid).sum(dim=1), valid.sum(dim=1)


def aligned_output_kl(clean_logits: torch.Tensor, changed_logits: torch.Tensor) -> torch.Tensor:
    clean_next = clean_logits[:, :-1, :].float()
    changed_next = changed_logits[:, 1:-1, :].float()
    clean_logp = clean_next.log_softmax(dim=-1)
    changed_logp = changed_next.log_softmax(dim=-1)
    clean_p = clean_logp.exp()
    return (clean_p * (clean_logp - changed_logp)).sum(dim=-1).mean(dim=1)


def mean_layer_attention(attentions: tuple[torch.Tensor, ...], *, bos_idx: int, dummy_idx: int, query_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    bos_by_layer = []
    dummy_by_layer = []
    for attn in attentions:
        a = attn.detach().float()
        bos_by_layer.append(a[:, :, query_start:, bos_idx].mean(dim=(1, 2)))
        dummy_by_layer.append(a[:, :, query_start:, dummy_idx].mean(dim=(1, 2)))
    bos = torch.stack(bos_by_layer, dim=1).mean(dim=1)
    dummy = torch.stack(dummy_by_layer, dim=1).mean(dim=1)
    return bos.cpu(), dummy.cpu()


@torch.no_grad()
def evaluate_condition(
    *,
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    condition: str,
    candidate_group: str,
    mode: str | None,
    scale: float,
    model_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for batch in bt.build_window_dataloader(windows, batch_size=bt.BATCH_SIZE):
        bsz = int(batch.shape[0])
        src_batch = source_mask[offset: offset + bsz]
        batch_indices = list(range(offset, offset + bsz))
        offset += bsz

        batch = batch.to(model.device)
        labels = batch.clone()
        labels[:, 0] = -100
        clean = model(input_ids=batch, use_cache=False, return_dict=True)
        clean_loss, clean_count = per_example_nll(clean.logits, labels)

        embeds, attention_mask = bt.build_intervened_inputs(model, batch, dummy_init="zero", num_dummy_tokens=1)
        if mode is None:
            changed = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        else:
            if bias_by_layer is None:
                raise ValueError(f"{condition} requires bias_by_layer")
            with bt.patched_bias_intervention(model, bias_by_layer, src_batch, scale, mode):
                changed = model(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
        ilabels = torch.full((batch.shape[0], batch.shape[1] + 1), -100, dtype=torch.long, device=batch.device)
        ilabels[:, 2:] = batch[:, 1:]
        changed_loss, changed_count = per_example_nll(changed.logits, ilabels)
        if not torch.equal(clean_count.cpu(), changed_count.cpu()):
            raise RuntimeError("token count mismatch")
        kl = aligned_output_kl(clean.logits, changed.logits).cpu()
        bos, dummy = mean_layer_attention(changed.attentions, bos_idx=0, dummy_idx=1, query_start=2)
        for local_idx, example_idx in enumerate(batch_indices):
            clean_nll = float(clean_loss[local_idx].item() / clean_count[local_idx].item())
            changed_nll = float(changed_loss[local_idx].item() / changed_count[local_idx].item())
            rows.append({
                "model_key": model_key,
                "condition": condition,
                "candidate_group": candidate_group,
                "example_idx": int(example_idx),
                "mode": mode or "blank_dummy",
                "scale": float(scale) if mode is not None else 0.0,
                "clean_nll": clean_nll,
                "intervened_nll": changed_nll,
                "delta_nll": changed_nll - clean_nll,
                "output_kl": float(kl[local_idx].item()),
                "bos_attention": float(bos[local_idx].item()),
                "dummy_attention": float(dummy[local_idx].item()),
                "dummy_minus_bos": float((dummy[local_idx] - bos[local_idx]).item()),
                "tokens": int(clean_count[local_idx].item()),
            })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    out: list[dict[str, Any]] = []
    for condition, group in sorted(by_condition.items()):
        tokens = sum(int(r["tokens"]) for r in group)
        clean_loss = sum(float(r["clean_nll"]) * int(r["tokens"]) for r in group)
        changed_loss = sum(float(r["intervened_nll"]) * int(r["tokens"]) for r in group)
        clean_nll = clean_loss / tokens
        changed_nll = changed_loss / tokens
        out.append({
            "condition": condition,
            "candidate_group": group[0]["candidate_group"],
            "mode": group[0]["mode"],
            "scale": group[0]["scale"],
            "n_examples": len(group),
            "tokens": tokens,
            "bos_attention_mean": sum(float(r["bos_attention"]) for r in group) / len(group),
            "dummy_attention_mean": sum(float(r["dummy_attention"]) for r in group) / len(group),
            "dummy_minus_bos_mean": sum(float(r["dummy_minus_bos"]) for r in group) / len(group),
            "output_kl_mean": sum(float(r["output_kl"]) for r in group) / len(group),
            "delta_nll_mean": changed_nll - clean_nll,
            "baseline_ppl": math.exp(clean_nll),
            "intervened_ppl": math.exp(changed_nll),
            "ppl_ratio": math.exp(changed_nll - clean_nll),
        })
    return out


def git_snapshot() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_readme(path: Path, *, args: argparse.Namespace, accepted: dict[str, str], cfg: Any, selected_path: Path, random_path: Path, command: str, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Intervention decomposition: {args.model_key}",
        "",
        "Status: complete.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        f"Script path: `run_intervention_decomposition.py`",
        f"Code snapshot: `{git_snapshot()}`",
        f"Model revision/id: `{cfg.model_id}`",
        f"Data split: Wikitext train windows for bias estimation, Wikitext test windows for reported decomposition; windows from `{args.root / args.model_key / ('windows_192x' + str(cfg.window_length) + '.pt')}`.",
        "Seeds: not a training job; random layer-matched control uses fixed seed "
        f"`{args.seed}`.",
        f"Neuron-list path: selected `{selected_path}`, random control `{random_path}`.",
        f"Output path: `{path}`.",
        f"Accepted relocation row: mask `{accepted['mask']}`, topk `{accepted['topk']}`, scale `{accepted['scale']}`, layers `{accepted['layers']}`.",
        "",
        "## Sanity Checks",
        "",
        "- Raw per-example rows are written in `per_example.csv`; aggregate condition rows are written in `aggregate_summary.csv`.",
        "- Each condition inserts exactly one zero dummy slot after BOS; only one bias intervention mode is active per condition.",
        "- Random controls are sampled from the same candidate artifact with layer counts matched to the selected neurons.",
        "- Output KL is computed on next-token logits aligned between clean and dummy-inserted sequences.",
    ]
    path.joinpath("README.md").write_text("\n".join(lines) + "\n")


def plot_summary(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(r["condition"]) for r in summary_rows]
    reloc = [float(r["dummy_minus_bos_mean"]) for r in summary_rows]
    ppl = [float(r["ppl_ratio"]) for r in summary_rows]
    kl = [float(r["output_kl_mean"]) for r in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, vals, title in zip(axes, [reloc, ppl, kl], ["dummy - BOS", "PPL ratio", "output KL"]):
        ax.bar(range(len(labels)), vals, color="#4C78A8")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.axhline(0 if title != "PPL ratio" else 1, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "intervention_decomposition_summary.png", dpi=180)
    fig.savefig(out_dir / "intervention_decomposition_summary.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run component decomposition for accepted final-split bias-transfer relocation rows.")
    parser.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs/final_split_bias_transfer"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-test-windows", type=int, default=64)
    parser.add_argument("--command", type=str, default="")
    args = parser.parse_args()

    print("[startup] ensuring runtime imports", flush=True)
    bt._ensure_runtime_imports()
    print("[startup] configuring runtime", flush=True)
    bt.configure_runtime()
    if args.out.exists():
        existing = [p for p in args.out.iterdir() if p.name != "logs"]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = cfg_for_key(args.model_key)
    model_dir = args.root / args.model_key
    accepted = read_first_row(model_dir / "test_result.csv")
    mask_name = str(accepted["mask"])
    topk = int(accepted["topk"])
    scale = float(accepted["scale"])
    candidate_path = model_dir / mask_name / "candidate_scores.pt"
    payload = bt.load_pt(candidate_path)
    selected = payload["selected_top"][:topk].clone().long()
    random_selected = sample_layer_matched_random(payload["score_rows"], selected, seed=args.seed)
    bt.save_pt(args.out / "selected_neurons.pt", {"selected": selected, "source": str(candidate_path), "topk": topk})
    bt.save_pt(args.out / "layer_matched_random_neurons.pt", {"selected": random_selected, "source": str(candidate_path), "seed": args.seed})

    windows_payload = bt.load_pt(model_dir / f"windows_192x{cfg.window_length}.pt")
    windows = windows_payload["windows"].long()
    splits = bt.split_windows(windows, bt.WINDOWS_PER_SPLIT)
    train = splits["train"]
    test = splits["test"][: args.max_test_windows]
    train_source = bt.load_pt(model_dir / "train_sink_mask.pt")["sink_mask"].bool()
    test_source = bt.load_pt(model_dir / "test_sink_mask.pt")["sink_mask"].bool()[: args.max_test_windows]
    train_mask = bt.make_masks(train, train_source)[mask_name]
    test_mask = bt.make_masks(test, test_source)[mask_name]

    print(f"[startup] loading model {cfg.model_id}", flush=True)
    model = bt.load_model(cfg.model_id, eager_attention=cfg.eager_attention)
    print("[startup] estimating selected bias", flush=True)
    bias_selected = bt.estimate_bias(model, train, selected, train_mask)
    print("[startup] estimating layer-matched random bias", flush=True)
    bias_random = bt.estimate_bias(model, train, random_selected, train_mask)

    all_rows: list[dict[str, Any]] = []
    for condition, candidate_group, mode in CONDITIONS:
        bias = None
        if candidate_group == "selected":
            bias = bias_selected
        elif candidate_group == "layer_matched_random":
            bias = bias_random
        rows = evaluate_condition(
            model=model,
            windows=test,
            source_mask=test_mask,
            bias_by_layer=bias,
            condition=condition,
            candidate_group=candidate_group,
            mode=mode,
            scale=scale,
            model_key=args.model_key,
        )
        all_rows.extend(rows)
        write_csv(args.out / "per_example.csv", all_rows)
        print(f"[done-condition] {condition}: n={len(rows)}", flush=True)

    summary_rows = summarize(all_rows)
    for row in summary_rows:
        row.update({
            "model_key": args.model_key,
            "model_id": cfg.model_id,
            "mask": mask_name,
            "topk": topk,
            "selected_layers": layer_counts(selected),
            "random_layers": layer_counts(random_selected),
            "candidate_path": str(candidate_path),
        })
    write_csv(args.out / "aggregate_summary.csv", summary_rows)
    with (args.out / "aggregate_summary.md").open("w") as handle:
        handle.write(f"# Intervention Decomposition Summary: {args.model_key}\n\n")
        handle.write("| condition | group | reloc | PPL ratio | KL | BOS attn | dummy attn |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['condition']} | {row['candidate_group']} | {row['dummy_minus_bos_mean']:.6g} | "
                f"{row['ppl_ratio']:.6g} | {row['output_kl_mean']:.6g} | {row['bos_attention_mean']:.6g} | "
                f"{row['dummy_attention_mean']:.6g} |\n"
            )
    plot_summary(args.out, summary_rows)
    write_readme(
        args.out,
        args=args,
        accepted=accepted,
        cfg=cfg,
        selected_path=args.out / "selected_neurons.pt",
        random_path=args.out / "layer_matched_random_neurons.pt",
        command=args.command or " ".join(os.sys.argv),
        summary_rows=summary_rows,
    )
    print(args.out / "aggregate_summary.md")


if __name__ == "__main__":
    main()
