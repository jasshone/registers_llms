
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface/hub")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset

import run_autoresearch_sink_mechanism as sink_mech
from run_autoresearch_sink_mechanism import estimate_bias, layer_counts, make_masks, patched_bias_intervention
from sink_neurons.artifacts import load_pt
from sink_neurons.env import configure_runtime
from sink_neurons.intervention import build_intervened_inputs
from sink_neurons.modeling import get_transformer_layers, load_model, load_tokenizer
from sink_neurons.windows import resolve_start_token


FINAL_ROOT = Path("outputs/final_split_bias_transfer")
OFFICIAL_VAL = Path("outputs/final_split_bias_transfer_official_val/official_validation_results.csv")
OUT = Path("outputs/pile_val_sharded_transfer")
PILE_DATASET = "mmosbach/pile-validation"
PILE_SPLIT = "validation"
TRAIN_WINDOWS = 64

DEFAULT_SMALL_FIRST = [
    "pythia_70m",
    "opt_125m",
    "pythia_160m",
    "gpt2",
    "opt_350m",
    "pythia_410m",
    "gpt2_medium",
    "phi_1_5",
    "pythia_1b",
    "llama3_2_1b",
    "pythia_1_4b",
    "qwen3_1_7b",
    "phi_2",
    "pythia_2_8b",
    "llama3_2_3b",
    "phi3_mini_4k",
    "phi3_5_mini",
    "qwen3_4b",
    "gpt2_large",
    "mistral_7b_v0_1",
    "mistral_7b_v0_3",
    "gpt2_xl",
    "qwen3_8b",
    "llama3_8b",
    "pythia_6_9b",
    "mistral_nemo_12b",
    "qwen3_14b",
    "pythia_12b",
    "phi4",
    "qwen3_30b_a3b",
]


@dataclass
class EvalSums:
    clean_loss_sum: float = 0.0
    changed_loss_sum: float = 0.0
    token_count: int = 0
    example_count: int = 0
    total_windows_seen: int = 0
    selected_windows_seen: int = 0
    bos_before_sum: list[float] | None = None
    bos_after_sum: list[float] | None = None
    dummy_after_sum: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shardable full-Pile transfer eval for fixed WikiText-selected rows.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_SMALL_FIRST)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--rows-csv", type=Path, default=OFFICIAL_VAL)
    parser.add_argument("--pile-dataset", default=PILE_DATASET)
    parser.add_argument("--pile-split", default=PILE_SPLIT)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-selected-windows", type=int, default=None, help="Debug/partial run cap after sharding.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bias-batch-size", type=int, default=8)
    parser.add_argument("--loss-token-chunk", type=int, default=2048)
    parser.add_argument("--save-every-batches", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume-partial", action="store_true", help="Ignore partial.json checkpoints when shard_result.csv is not complete.")
    parser.add_argument("--resume-skip-log-every", type=int, default=25000, help="Print resume skip progress every N selected windows; <=0 disables.")
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["model_key"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "model_key", "model_id", "status", "shard_index", "num_shards", "selected_windows_seen", "total_windows_seen",
        "window_length", "mask", "topk", "scale", "candidate_group", "layers", "batch_size", "bias_batch_size",
        "bos_before_mean", "bos_after_mean", "dummy_after_mean", "dummy_minus_bos_mean", "baseline_ppl", "intervened_ppl",
        "ppl_ratio", "delta_mean_nll", "source_wikitext_dummy_minus_bos", "source_wikitext_ppl_ratio",
        "model_load_seconds", "bias_seconds", "eval_seconds", "eval_windows_per_second", "error",
    ]
    keys = [key for key in preferred if any(key in row for row in rows)]
    keys += [key for key in sorted({key for row in rows for key in row}) if key not in keys]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def window_length_for_model(model_id: str) -> int:
    return 1023 if model_id.startswith("gpt2") else 1024


def iter_dataset_texts(dataset_name: str, split: str, text_field: str) -> Iterator[str]:
    ds = load_dataset(dataset_name, split=split, streaming=True)
    for item in ds:
        text = item.get(text_field)
        if text:
            yield str(text)


def iter_sharded_windows(
    tokenizer: Any,
    *,
    dataset_name: str,
    split: str,
    text_field: str,
    window_length: int,
    stride: int,
    num_shards: int,
    shard_index: int,
    max_selected_windows: int | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    start_token = resolve_start_token(tokenizer)
    content_length = window_length - 1
    token_buffer: list[int] = []
    global_window_idx = 0
    selected = 0
    for text in iter_dataset_texts(dataset_name, split, text_field):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(int(x) for x in ids)
        while len(token_buffer) >= content_length:
            if global_window_idx % num_shards == shard_index:
                window = torch.tensor([start_token.token_id, *token_buffer[:content_length]], dtype=torch.long)
                yield global_window_idx, window
                selected += 1
                if max_selected_windows is not None and selected >= max_selected_windows:
                    return
            global_window_idx += 1
            drop = min(stride, len(token_buffer))
            token_buffer = token_buffer[drop:]


def load_train_windows(model_key: str, window_length: int) -> torch.Tensor:
    path = FINAL_ROOT / model_key / f"windows_{TRAIN_WINDOWS * 3}x{window_length}.pt"
    payload = load_pt(path)
    windows = payload.get("windows")
    if windows is None:
        windows = payload.get("tensors", {}).get("windows")
    if not isinstance(windows, torch.Tensor):
        raise ValueError(f"no windows tensor in {path}")
    return windows[:TRAIN_WINDOWS].long().clone()


def load_selected(model_key: str, row: dict[str, str]) -> torch.Tensor:
    topk = int(row["topk"])
    mask = row["mask"]
    group = row.get("candidate_group", "")
    if model_key == "pythia_2_8b" and group == "early_0_8":
        path = FINAL_ROOT / "pythia_2_8b_new_transfer_train_val_search" / "candidate_scores" / "full_val_clean_early_0_8_bos_only.pt"
        payload = load_pt(path)
        selected = payload.get("tensors", {}).get("selected_top")
        if selected is None:
            selected = payload.get("selected_top")
    elif group:
        path = FINAL_ROOT / model_key / f"rescue_scores_{group}_{mask}.json"
        if path.exists():
            selected = torch.tensor(json.loads(path.read_text())["selected_top"], dtype=torch.long)
        else:
            path = FINAL_ROOT / model_key / group / "candidate_scores.pt"
            payload = load_pt(path)
            selected = payload.get("selected_top")
            if selected is None:
                selected = payload.get("tensors", {}).get("selected_top")
    else:
        path = FINAL_ROOT / model_key / mask / "candidate_scores.pt"
        payload = load_pt(path)
        selected = payload.get("selected_top")
        if selected is None:
            selected = payload.get("tensors", {}).get("selected_top")
    if not isinstance(selected, torch.Tensor):
        raise ValueError(f"could not load selected neurons for {model_key}")
    selected = selected.cpu().long()[:topk].clone()
    if selected.numel() == 0:
        raise ValueError(f"empty selected neurons for {model_key}")
    return selected


def window_nll_chunked(logits: torch.Tensor, labels: torch.Tensor, *, token_chunk: int) -> tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    total_count = int(flat_labels.ne(-100).sum().item())
    total_loss = 0.0
    if token_chunk <= 0:
        token_chunk = int(flat_labels.numel())
    for start in range(0, int(flat_labels.numel()), token_chunk):
        end = min(start + token_chunk, int(flat_labels.numel()))
        chunk_labels = flat_labels[start:end]
        valid = chunk_labels.ne(-100)
        if bool(valid.any()):
            total_loss += float(F.cross_entropy(flat_logits[start:end][valid].float(), chunk_labels[valid], reduction="sum").item())
    return total_loss, total_count


def num_layers(model: torch.nn.Module) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    return len(get_transformer_layers(model))


def init_sums(n_layers: int) -> EvalSums:
    return EvalSums(
        bos_before_sum=[0.0 for _ in range(n_layers)],
        bos_after_sum=[0.0 for _ in range(n_layers)],
        dummy_after_sum=[0.0 for _ in range(n_layers)],
    )


def load_partial_checkpoint(path: Path, n_layers: int) -> tuple[EvalSums, float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    raw = payload.get("sums")
    if not isinstance(raw, dict):
        return None
    sums = EvalSums(**raw)
    if sums.bos_before_sum is None or len(sums.bos_before_sum) != n_layers:
        return None
    if sums.bos_after_sum is None or len(sums.bos_after_sum) != n_layers:
        return None
    if sums.dummy_after_sum is None or len(sums.dummy_after_sum) != n_layers:
        return None
    return sums, float(payload.get("elapsed_seconds", 0.0) or 0.0)


def save_partial(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def finalize_row(base: dict[str, Any], sums: EvalSums, eval_seconds: float) -> dict[str, Any]:
    examples = max(1, sums.example_count)
    bos_before = torch.tensor(sums.bos_before_sum or [], dtype=torch.float64) / examples
    bos_after = torch.tensor(sums.bos_after_sum or [], dtype=torch.float64) / examples
    dummy_after = torch.tensor(sums.dummy_after_sum or [], dtype=torch.float64) / examples
    clean_nll = sums.clean_loss_sum / max(1, sums.token_count)
    changed_nll = sums.changed_loss_sum / max(1, sums.token_count)
    return base | {
        "selected_windows_seen": sums.selected_windows_seen,
        "total_windows_seen": sums.total_windows_seen,
        "bos_before_mean": float(bos_before.mean().item()) if bos_before.numel() else float("nan"),
        "bos_after_mean": float(bos_after.mean().item()) if bos_after.numel() else float("nan"),
        "dummy_after_mean": float(dummy_after.mean().item()) if dummy_after.numel() else float("nan"),
        "dummy_minus_bos_mean": float((dummy_after - bos_after).mean().item()) if bos_after.numel() else float("nan"),
        "baseline_ppl": math.exp(clean_nll),
        "intervened_ppl": math.exp(changed_nll),
        "ppl_ratio": math.exp(changed_nll - clean_nll),
        "delta_mean_nll": changed_nll - clean_nll,
        "eval_seconds": eval_seconds,
        "eval_windows_per_second": sums.selected_windows_seen / eval_seconds if eval_seconds > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate_stream(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    model_key: str,
    model_id: str,
    window_length: int,
    source_mask_name: str,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    base_row: dict[str, Any],
    out_dir: Path,
    initial_sums: EvalSums | None = None,
    initial_elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    n_layers = num_layers(model)
    sums = initial_sums or init_sums(n_layers)
    resume_selected = int(sums.selected_windows_seen)
    skipped_selected = 0
    batch_windows: list[torch.Tensor] = []
    batch_indices: list[int] = []
    batches_done = 0
    t_eval = time.perf_counter() - float(initial_elapsed_seconds)

    def process_batch() -> None:
        nonlocal batches_done, sums
        if not batch_windows:
            return
        batch = torch.stack(batch_windows).to(model.device)
        bsz = int(batch.shape[0])
        source_mask = make_masks(batch.cpu(), torch.zeros_like(batch.cpu(), dtype=torch.bool))[source_mask_name]
        clean = model(input_ids=batch, output_attentions=True, use_cache=False, return_dict=True)
        labels = batch.clone()
        labels[:, 0] = -100
        loss, count = window_nll_chunked(clean.logits, labels, token_chunk=args.loss_token_chunk)
        sums.clean_loss_sum += loss
        sums.token_count += count
        with patched_bias_intervention(model, bias_by_layer, source_mask, scale, "transfer"):
            embeds, attn_mask = build_intervened_inputs(model, batch, dummy_init="zero", num_dummy_tokens=1)
            changed = model(inputs_embeds=embeds, attention_mask=attn_mask, output_attentions=True, use_cache=False, return_dict=True)
        ilabels = torch.full((batch.shape[0], batch.shape[1] + 1), -100, dtype=torch.long, device=batch.device)
        ilabels[:, 2:] = batch[:, 1:]
        iloss, icount = window_nll_chunked(changed.logits, ilabels, token_chunk=args.loss_token_chunk)
        if icount != count:
            raise RuntimeError("token mismatch")
        sums.changed_loss_sum += iloss
        sums.example_count += bsz
        sums.selected_windows_seen += bsz
        sums.total_windows_seen = max(sums.total_windows_seen, max(batch_indices) + 1)
        for layer_idx, attn in enumerate(clean.attentions):
            sums.bos_before_sum[layer_idx] += attn[:, :, 1:, 0].float().mean(dim=(1, 2)).sum().item()
        for layer_idx, attn in enumerate(changed.attentions):
            sums.bos_after_sum[layer_idx] += attn[:, :, 2:, 0].float().mean(dim=(1, 2)).sum().item()
            sums.dummy_after_sum[layer_idx] += attn[:, :, 2:, 1].float().mean(dim=(1, 2)).sum().item()
        batches_done += 1
        batch_windows.clear()
        batch_indices.clear()
        if args.save_every_batches > 0 and batches_done % args.save_every_batches == 0:
            elapsed = time.perf_counter() - t_eval
            save_partial(out_dir / "partial.json", {"base": base_row, "sums": asdict(sums), "elapsed_seconds": elapsed})
            write_csv(out_dir / "shard_result.csv", [finalize_row(base_row | {"status": "partial"}, sums, elapsed)])

    for global_idx, window in iter_sharded_windows(
        tokenizer,
        dataset_name=args.pile_dataset,
        split=args.pile_split,
        text_field=args.text_field,
        window_length=window_length,
        stride=window_length - 1,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_selected_windows=args.max_selected_windows,
    ):
        if skipped_selected < resume_selected:
            skipped_selected += 1
            if args.resume_skip_log_every > 0 and skipped_selected % args.resume_skip_log_every == 0:
                print(f"[resume-skip] {model_key}: {skipped_selected}/{resume_selected}", flush=True)
            continue
        if resume_selected and skipped_selected == resume_selected:
            print(f"[resume-skip] {model_key}: reached {resume_selected}; resuming eval", flush=True)
            skipped_selected += 1
        batch_windows.append(window)
        batch_indices.append(global_idx)
        if len(batch_windows) >= args.batch_size:
            process_batch()
    process_batch()
    eval_seconds = time.perf_counter() - t_eval
    row = finalize_row(base_row | {"status": "ok"}, sums, eval_seconds)
    save_partial(out_dir / "partial.json", {"base": base_row, "sums": asdict(sums), "elapsed_seconds": eval_seconds})
    write_csv(out_dir / "shard_result.csv", [row])
    return row


def run_one(args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    model_key = row["model_key"]
    model_id = row["model_id"]
    mask_name = row["mask"]
    if mask_name != "bos_only":
        raise ValueError(f"only bos_only rows are supported in this first sharded runner, got {mask_name}")
    window_length = window_length_for_model(model_id)
    out_dir = args.out / model_key / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    result_path = out_dir / "shard_result.csv"
    if result_path.exists() and not args.force:
        existing = list(csv.DictReader(result_path.open()))
        if existing and existing[0].get("status") == "ok":
            print(f"[skip] {model_key}: existing complete shard", flush=True)
            return existing[0]

    tokenizer = load_tokenizer(model_id)
    train_windows = load_train_windows(model_key, window_length)
    selected = load_selected(model_key, row)
    t_model = time.perf_counter()
    model = load_model(model_id, eager_attention=True)
    model_load_seconds = time.perf_counter() - t_model
    train_mask = make_masks(train_windows, torch.zeros_like(train_windows, dtype=torch.bool))[mask_name]
    old_batch = sink_mech.BATCH_SIZE
    sink_mech.BATCH_SIZE = args.bias_batch_size
    t_bias = time.perf_counter()
    try:
        bias_by_layer = estimate_bias(model, train_windows, selected, train_mask)
    finally:
        sink_mech.BATCH_SIZE = old_batch
    bias_seconds = time.perf_counter() - t_bias

    base = {
        "model_key": model_key,
        "model_id": model_id,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "eval_dataset": args.pile_dataset,
        "eval_split": args.pile_split,
        "window_length": window_length,
        "mask": mask_name,
        "topk": int(row["topk"]),
        "scale": float(row["scale"]),
        "candidate_group": row.get("candidate_group", ""),
        "layers": layer_counts(selected),
        "batch_size": args.batch_size,
        "bias_batch_size": args.bias_batch_size,
        "source_wikitext_dummy_minus_bos": row.get("dummy_minus_bos_mean"),
        "source_wikitext_ppl_ratio": row.get("ppl_ratio"),
        "model_load_seconds": model_load_seconds,
        "bias_seconds": bias_seconds,
    }
    initial_sums = None
    initial_elapsed_seconds = 0.0
    if not args.force and not args.no_resume_partial:
        checkpoint = load_partial_checkpoint(out_dir / "partial.json", num_layers(model))
        if checkpoint is not None:
            initial_sums, initial_elapsed_seconds = checkpoint
            if initial_sums.selected_windows_seen:
                print(f"[resume] {model_key}: skipping {initial_sums.selected_windows_seen} completed shard windows", flush=True)

    result = evaluate_stream(
        args=args,
        model=model,
        tokenizer=tokenizer,
        model_key=model_key,
        model_id=model_id,
        window_length=window_length,
        source_mask_name=mask_name,
        bias_by_layer=bias_by_layer,
        scale=float(row["scale"]),
        base_row=base,
        out_dir=out_dir,
        initial_sums=initial_sums,
        initial_elapsed_seconds=initial_elapsed_seconds,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def aggregate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(args.out.glob("*/shard_*_of_*/shard_result.csv")):
        with path.open() as handle:
            shard_rows = list(csv.DictReader(handle))
        if shard_rows:
            rows.append(shard_rows[0])
    write_csv(args.out / "summary_shards.csv", rows)
    return rows


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    configure_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    rows_by_model = read_rows(args.rows_csv)
    run_rows = []
    for model_key in args.models:
        print(f"[start] {model_key} shard {args.shard_index}/{args.num_shards}", flush=True)
        try:
            result = run_one(args, rows_by_model[model_key])
            print(
                f"[done] {model_key}: windows={result.get('selected_windows_seen')} "
                f"reloc={float(result['dummy_minus_bos_mean']):.6g} ppl={float(result['ppl_ratio']):.6g} "
                f"wps={float(result['eval_windows_per_second']):.3f}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "model_key": model_key,
                "status": "error",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "error": repr(exc),
            }
            err_dir = args.out / model_key / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
            write_csv(err_dir / "shard_result.csv", [result])
            print(f"[error] {model_key}: {exc!r}", flush=True)
        run_rows.append(result)
        aggregate(args)
    write_csv(args.out / f"run_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.csv", run_rows)
    aggregate(args)


if __name__ == "__main__":
    main()
