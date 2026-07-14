from __future__ import annotations

import argparse
import csv
import gc
import os
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/registers_llms/.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("TMPDIR", "/workspace/registers_llms/.tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from sink_neurons.artifacts import load_pt
import sink_neurons.bias_transfer as bt
from sink_neurons.bias_transfer import CONFIGS, evaluate_with_fixed_bias
from sink_neurons.causal_diagnostics import layer_counts
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import load_model

DEFAULT_MODELS = ["gpt2_medium", "pythia_1b", "qwen3_1_7b", "llama3_2_1b", "mistral_7b_v0_1", "phi_2"]
CONFIG_BY_KEY = {cfg.key: cfg for cfg in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate accepted representative transfer rows on in-sample train windows.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--root", type=Path, default=Path("outputs/final_split_bias_transfer"))
    parser.add_argument("--out", type=Path, default=Path("outputs/representative_train_transfer_eval"))
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row.get("model_key"))))


def load_windows(root: Path, model_key: str, cfg: Any, count: int) -> torch.Tensor:
    path = root / model_key / f"windows_192x{cfg.window_length}.pt"
    payload = load_pt(path)
    windows = payload.get("windows")
    if windows is None:
        windows = payload.get("tensors", {}).get("windows")
    if not isinstance(windows, torch.Tensor):
        raise ValueError(f"no windows tensor in {path}")
    if int(windows.shape[0]) < count:
        raise ValueError(f"need {count} train windows, found {windows.shape[0]}")
    return windows[:count].long().clone()


def load_sink_mask(root: Path, model_key: str, windows: torch.Tensor) -> torch.Tensor:
    for name in ("train_attention_sink_mask.pt", "train_sink_mask.pt"):
        path = root / model_key / name
        if not path.exists():
            continue
        payload = load_pt(path)
        mask = payload.get("sink_mask")
        if mask is None:
            mask = payload.get("tensors", {}).get("sink_mask")
        if isinstance(mask, torch.Tensor):
            if tuple(mask.shape) != tuple(windows.shape):
                return mask[: windows.shape[0], : windows.shape[1]].bool().clone()
            return mask.bool().clone()
    raise FileNotFoundError(f"no train sink mask for {model_key}")


def load_selected(root: Path, model_key: str, row: dict[str, str]) -> torch.Tensor:
    mask = row["mask"]
    topk = int(float(row["topk"]))
    payload = load_pt(root / model_key / mask / "candidate_scores.pt")
    selected = payload.get("selected_top")
    if selected is None:
        selected = payload.get("tensors", {}).get("selected_top")
    if not isinstance(selected, torch.Tensor):
        raise ValueError(f"no selected_top for {model_key}/{mask}")
    return selected[:topk].long().clone()


def run_one(args: argparse.Namespace, model_key: str) -> dict[str, Any]:
    cfg = CONFIG_BY_KEY[model_key]
    accepted_rows = [row for row in read_csv(args.root / model_key / "test_result.csv") if row.get("status") == "ok"]
    if not accepted_rows:
        raise ValueError(f"no accepted test row for {model_key}")
    accepted = accepted_rows[0]
    windows = load_windows(args.root, model_key, cfg, int(args.windows))
    train_sink = load_sink_mask(args.root, model_key, windows)
    mask_name = accepted["mask"]
    train_masks = bt.make_masks(windows, train_sink)
    if mask_name not in train_masks:
        raise ValueError(f"mask {mask_name} not available for {model_key}")
    source_mask = train_masks[mask_name]
    selected = load_selected(args.root, model_key, accepted)
    model = load_model(cfg.model_id, eager_attention=cfg.eager_attention)
    try:
        bias_by_layer = bt.estimate_bias(model, windows, selected, source_mask)
        transfer = evaluate_with_fixed_bias(model, windows, selected, source_mask, bias_by_layer, "transfer", float(accepted["scale"]))
        subtract = evaluate_with_fixed_bias(model, windows, selected, source_mask, bias_by_layer, "subtract_only", 1.0)
        add = evaluate_with_fixed_bias(model, windows, selected, source_mask, bias_by_layer, "add_only", 1.0)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row: dict[str, Any] = {
        "status": "ok",
        "family": cfg.family,
        "model_key": model_key,
        "model_id": cfg.model_id,
        "split": "train_stream_train_in_sample",
        "windows": int(windows.shape[0]),
        "mask": mask_name,
        "topk": int(float(accepted["topk"])),
        "scale": float(accepted["scale"]),
        "layers": layer_counts(selected),
        "note": "in-sample: train windows are the selection/scoring split",
    }
    for prefix, result in (("transfer", transfer), ("subtract_only", subtract), ("add_only", add)):
        for key, value in result.items():
            row[f"{prefix}_{key}"] = value
    return row


def fmt(row: dict[str, Any], key: str) -> str:
    try:
        return f"{float(row[key]):.4g}"
    except Exception:
        return ""


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# Representative Train Transfer Evaluation",
        "",
        "Accepted final split transfer rows evaluated on the train-window split. These are in-sample rows because train was used for candidate scoring and bias estimation; use validation/test as the main held-out evidence.",
        "",
        "| model | mask | topk | scale | train transfer reloc | train transfer PPL | subtract-only PPL | add-only reloc | add-only PPL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(ok, key=lambda row: str(row["model_key"])):
        lines.append(
            f"| `{row['model_key']}` | `{row['mask']}` | {row['topk']} | {float(row['scale']):g} | "
            f"{fmt(row, 'transfer_dummy_minus_bos_mean')} | {fmt(row, 'transfer_ppl_ratio')} | "
            f"{fmt(row, 'subtract_only_ppl_ratio')} | {fmt(row, 'add_only_dummy_minus_bos_mean')} | {fmt(row, 'add_only_ppl_ratio')} |"
        )
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        for row in errors:
            lines.append(f"- `{row.get('model_key')}`: {row.get('error')}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    configure_runtime()
    bt._ensure_runtime_imports()
    args = parse_args()
    out_dir = args.out / f"train_{args.windows}w"
    csv_path = out_dir / "representative_train_transfer_eval.csv"
    md_path = out_dir / "representative_train_transfer_eval.md"
    rows = [] if args.force else read_csv(csv_path)  # type: ignore[assignment]
    done = {str(row.get("model_key")) for row in rows if row.get("status") == "ok"}
    for model_key in args.models:
        if model_key in done:
            print(f"[skip] {model_key}", flush=True)
            continue
        print(f"[run] {model_key}", flush=True)
        try:
            row = run_one(args, model_key)
            print(f"[row] {model_key}: train_reloc={row['transfer_dummy_minus_bos_mean']:.4g} ppl={row['transfer_ppl_ratio']:.4g}", flush=True)
        except Exception as exc:
            row = {"status": "error", "model_key": model_key, "error": repr(exc), "traceback": traceback.format_exc()}
            print(f"[error] {model_key}: {exc!r}", flush=True)
        rows = [old for old in rows if str(old.get("model_key")) != model_key]
        rows.append(row)
        write_csv(csv_path, rows)
        write_md(md_path, rows)
    print(md_path)


if __name__ == "__main__":
    main()
