from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

import sink_neurons.bias_transfer as bt

DEFAULT_WINDOWS = Path("outputs/final_split_bias_transfer_source_abs_bonus0_official_scope/pythia_1b/windows_192x1024.pt")
DEFAULT_OUT = Path("outputs/split_half_selection_20260715/pythia_1b")
MODEL_ID = "EleutherAI/pythia-1b"
CANDIDATE_LAYERS = [1, 2, 3, 4, 5, 6]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_windows(path: Path) -> torch.Tensor:
    payload = bt.load_pt(path)
    if isinstance(payload.get("tensors"), dict) and "windows" in payload["tensors"]:
        return payload["tensors"]["windows"].long().cpu()
    if "windows" in payload:
        return payload["windows"].long().cpu()
    raise KeyError(path)


def bos_mask(windows: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(windows, dtype=torch.bool)
    mask[:, 0] = True
    return mask


def layer_counts(selected: torch.Tensor) -> str:
    if selected.numel() == 0:
        return ""
    parts = []
    for layer in selected[:, 0].unique(sorted=True).tolist():
        parts.append(f"{int(layer)}:{int((selected[:, 0] == int(layer)).sum().item())}")
    return " ".join(parts)


def selection_set(selected: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(layer), int(neuron)) for layer, neuron in selected.cpu().tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pythia-1B split-half BOS-only neuron reselection and cross-evaluation.")
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--half-windows", type=int, default=64)
    args = parser.parse_args()

    bt._ensure_runtime_imports()
    args.out.mkdir(parents=True, exist_ok=True)
    windows = load_windows(args.windows)
    n = int(args.half_windows)
    if windows.shape[0] < 3 * n:
        raise ValueError(f"need at least {3*n} windows, got {windows.shape[0]}")
    splits = {
        "A": windows[:n].clone(),
        "B": windows[n:2*n].clone(),
        "test": windows[2*n:3*n].clone(),
    }
    masks = {name: bos_mask(tensor) for name, tensor in splits.items()}
    model = bt.load_model(MODEL_ID, eager_attention=True)

    selections: dict[str, torch.Tensor] = {}
    biases: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    score_paths: dict[str, str] = {}
    for name in ["A", "B"]:
        selected_top, score_rows = bt.score_candidates(
            model,
            splits[name],
            masks[name],
            CANDIDATE_LAYERS,
            max_top=int(args.topk),
            source_abs_bonus=0.0,
        )
        selected = selected_top[: int(args.topk)].cpu().long().clone()
        selections[name] = selected
        biases[name] = bt.estimate_bias(model, splits[name], selected, masks[name])
        score_path = args.out / f"score_rows_{name}.csv"
        write_csv(score_path, [
            {"score": a, "layer": b, "neuron": c, "gap": d, "source_mean": e, "control_mean": f}
            for a, b, c, d, e, f in score_rows
        ])
        score_paths[name] = str(score_path)
        bt.save_pt(args.out / f"selection_{name}.pt", {"selected": selected, "source": name, "topk": int(args.topk), "candidate_layers": CANDIDATE_LAYERS})

    set_a = selection_set(selections["A"])
    set_b = selection_set(selections["B"])
    overlap = len(set_a & set_b)
    rows: list[dict[str, Any]] = []
    for source in ["A", "B"]:
        for eval_split in ["A", "B", "test"]:
            result = bt.evaluate_with_fixed_bias(
                model,
                splits[eval_split],
                selections[source],
                masks[eval_split],
                biases[source],
                "transfer",
                float(args.scale),
            )
            rows.append({
                "model_key": "pythia_1b",
                "selection_source": source,
                "eval_split": eval_split,
                "cross_eval": source != eval_split,
                "topk": int(args.topk),
                "scale": float(args.scale),
                "mask": "bos_only",
                "candidate_layers": " ".join(str(x) for x in CANDIDATE_LAYERS),
                "layers": layer_counts(selections[source]),
                "selection_overlap_with_other_half": overlap,
                "selection_jaccard": overlap / max(1, len(set_a | set_b)),
                **result,
            })
    write_csv(args.out / "split_half_cross_eval.csv", rows)
    summary = {
        "model_key": "pythia_1b",
        "model_id": MODEL_ID,
        "windows": str(args.windows),
        "half_windows": int(args.half_windows),
        "topk": int(args.topk),
        "scale": float(args.scale),
        "candidate_layers": CANDIDATE_LAYERS,
        "source_mask": "bos_only",
        "selection_overlap": overlap,
        "selection_union": len(set_a | set_b),
        "selection_jaccard": overlap / max(1, len(set_a | set_b)),
        "score_paths": score_paths,
    }
    (args.out / "split_half_config.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Split-Half Neuron Selection Robustness",
        "",
        "Pythia-1B BOS-only source-mask check. Two disjoint 64-window Wikitext halves select top-16 neurons independently; each selected set is cross-evaluated on the other half and on the held-out third block using the fixed scale 2.0 transfer intervention.",
        "",
        f"Selection overlap: `{overlap}/{int(args.topk)}`; Jaccard `{summary['selection_jaccard']:.6g}`.",
        "",
        "| selection source | eval split | cross? | dummy-BOS | PPL ratio | BOS after | dummy after | layers |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {selection_source} | {eval_split} | {cross_eval} | {dummy_minus_bos_mean:.6g} | {ppl_ratio:.6g} | {bos_after_mean:.6g} | {dummy_after_mean:.6g} | `{layers}` |".format(**row)
        )
    lines.extend([
        "",
        "Interpretation: this is an appendix robustness check, not a full replacement for frozen Wikitext-to-Pile/MMLU transfer. Positive cross-eval dummy-minus-BOS indicates the reselection is not tied to a single train subset.",
    ])
    (args.out / "split_half_summary.md").write_text("\n".join(lines) + "\n")
    print(args.out / "split_half_summary.md")


if __name__ == "__main__":
    main()
