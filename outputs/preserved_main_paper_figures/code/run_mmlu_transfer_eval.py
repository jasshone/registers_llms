from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

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
OUT = Path("outputs/mmlu_transfer_eval")
TRAIN_WINDOWS = 64
DEFAULT_MODELS = [
    "gpt2_medium",
    "opt_350m",
    "pythia_2_8b",
    "phi_2",
    "llama3_2_1b",
    "qwen3_1_7b",
    "mistral_7b_v0_1",
]
CHOICE_LABELS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed WikiText-selected transfer rows on MMLU.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--rows-csv", type=Path, default=OFFICIAL_VAL)
    parser.add_argument("--dataset", default="cais/mmlu")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--sample-mode", choices=["head", "stratified"], default="stratified")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject subset after loading.")
    parser.add_argument("--bias-batch-size", type=int, default=8)
    parser.add_argument("--loss-token-chunk", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Original sequence cap before dummy insertion.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-intervened-accuracy", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["model_key"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "model_key", "model_id", "status", "eval_dataset", "eval_config", "eval_split", "eval_examples",
        "subjects", "mask", "topk", "scale", "candidate_group", "layers", "bias_batch_size",
        "clean_accuracy", "intervened_accuracy", "accuracy_delta", "clean_mean_choice_nll",
        "intervened_mean_choice_nll", "bos_before_mean", "bos_after_mean", "dummy_after_mean",
        "dummy_minus_bos_mean", "source_wikitext_dummy_minus_bos", "source_wikitext_ppl_ratio",
        "model_load_seconds", "bias_seconds", "eval_seconds", "examples_per_second", "forwards_per_second",
        "total_forwards", "max_original_seq_len", "error",
    ]
    keys = [key for key in preferred if any(key in row for row in rows)]
    keys += [key for key in sorted({key for row in rows for key in row}) if key not in keys]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def window_length_for_model(model_id: str) -> int:
    return 1023 if model_id.startswith("gpt2") else 1024


def model_context_limit(model: torch.nn.Module, requested: int | None) -> int:
    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(model.config, "seq_length", None),
    ]
    limit = min([int(x) for x in candidates if x is not None] or [2048])
    if requested is not None:
        limit = min(limit, int(requested))
    return max(8, limit - 1)


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


def num_layers(model: torch.nn.Module) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    return len(get_transformer_layers(model))


def load_mmlu_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    ds = load_dataset(args.dataset, args.config, split=args.split)
    rows: list[dict[str, Any]] = []
    subjects = set(args.subjects or [])
    for item in ds:
        subject = str(item.get("subject", ""))
        if subjects and subject not in subjects:
            continue
        choices = list(item["choices"])
        if len(choices) != 4:
            continue
        rows.append({
            "question": str(item["question"]),
            "subject": subject,
            "choices": [str(choice) for choice in choices],
            "answer": int(item["answer"]),
        })
        if args.sample_mode == "head" and args.max_examples is not None and len(rows) >= args.max_examples:
            break
    if not rows:
        raise ValueError("MMLU selection produced no examples")
    if args.sample_mode == "head" or args.max_examples is None or len(rows) <= args.max_examples:
        return rows[: args.max_examples]

    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    selected: list[dict[str, Any]] = []
    subject_order = sorted(by_subject)
    depth = 0
    while len(selected) < args.max_examples:
        added = False
        for subject in subject_order:
            bucket = by_subject[subject]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) >= args.max_examples:
                    break
        if not added:
            break
        depth += 1
    return selected


def format_prompt(example: dict[str, Any]) -> str:
    lines = [f"Question: {example['question']}"]
    for label, choice in zip(CHOICE_LABELS, example["choices"]):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def encode_prompt(tokenizer: Any, start_token_id: int, prompt: str, answer: str | None, max_original_len: int) -> tuple[torch.Tensor, int]:
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    answer_ids: list[int] = []
    if answer is not None:
        answer_ids = list(tokenizer(" " + answer, add_special_tokens=False)["input_ids"])
        if not answer_ids:
            answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    room_for_prompt = max(1, max_original_len - 1 - len(answer_ids))
    if len(prompt_ids) > room_for_prompt:
        prompt_ids = prompt_ids[-room_for_prompt:]
    ids = [int(start_token_id), *[int(x) for x in prompt_ids], *[int(x) for x in answer_ids]]
    answer_start = 1 + len(prompt_ids)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0), answer_start


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor, *, token_chunk: int) -> tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    count = int(flat_labels.ne(-100).sum().item())
    loss = 0.0
    if token_chunk <= 0:
        token_chunk = int(flat_labels.numel())
    for start in range(0, int(flat_labels.numel()), token_chunk):
        end = min(start + token_chunk, int(flat_labels.numel()))
        chunk_labels = flat_labels[start:end]
        valid = chunk_labels.ne(-100)
        if bool(valid.any()):
            loss += float(F.cross_entropy(flat_logits[start:end][valid].float(), chunk_labels[valid], reduction="sum").item())
    return loss, count


@torch.no_grad()
def score_choice(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    answer_start: int,
    *,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    scale: float,
    token_chunk: int,
) -> tuple[float, int]:
    input_ids = input_ids.to(model.device)
    if bias_by_layer is None:
        out = model(input_ids=input_ids, output_attentions=False, use_cache=False, return_dict=True)
        labels = torch.full_like(input_ids, -100)
        labels[:, answer_start:] = input_ids[:, answer_start:]
    else:
        source_mask = torch.zeros_like(input_ids, dtype=torch.bool, device="cpu")
        source_mask[:, 0] = True
        with patched_bias_intervention(model, bias_by_layer, source_mask, scale, "transfer"):
            embeds, attention_mask = build_intervened_inputs(model, input_ids, dummy_init="zero", num_dummy_tokens=1)
            out = model(inputs_embeds=embeds, attention_mask=attention_mask, output_attentions=False, use_cache=False, return_dict=True)
        labels = torch.full((1, input_ids.shape[1] + 1), -100, dtype=torch.long, device=input_ids.device)
        labels[:, answer_start + 1:] = input_ids[:, answer_start:]
    return sequence_nll(out.logits, labels, token_chunk=token_chunk)


@torch.no_grad()
def relocation_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> tuple[list[float], list[float], list[float]]:
    input_ids = input_ids.to(model.device)
    clean = model(input_ids=input_ids, output_attentions=True, use_cache=False, return_dict=True)
    source_mask = torch.zeros_like(input_ids, dtype=torch.bool, device="cpu")
    source_mask[:, 0] = True
    with patched_bias_intervention(model, bias_by_layer, source_mask, scale, "transfer"):
        embeds, attention_mask = build_intervened_inputs(model, input_ids, dummy_init="zero", num_dummy_tokens=1)
        changed = model(inputs_embeds=embeds, attention_mask=attention_mask, output_attentions=True, use_cache=False, return_dict=True)
    bos_before: list[float] = []
    bos_after: list[float] = []
    dummy_after: list[float] = []
    for attn in clean.attentions:
        bos_before.append(float(attn[:, :, 1:, 0].float().mean().item()))
    for attn in changed.attentions:
        bos_after.append(float(attn[:, :, 2:, 0].float().mean().item()))
        dummy_after.append(float(attn[:, :, 2:, 1].float().mean().item()))
    return bos_before, bos_after, dummy_after


@torch.no_grad()
def evaluate_mmlu(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    max_original_len: int,
    token_chunk: int,
    *,
    intervened_accuracy: bool,
) -> dict[str, float]:
    start_token = resolve_start_token(tokenizer)
    n_layers = num_layers(model)
    bos_before = torch.zeros(n_layers, dtype=torch.float64)
    bos_after = torch.zeros(n_layers, dtype=torch.float64)
    dummy_after = torch.zeros(n_layers, dtype=torch.float64)
    clean_correct = 0
    changed_correct = 0
    clean_loss = 0.0
    changed_loss = 0.0
    clean_tokens = 0
    changed_tokens = 0
    total_forwards = 0

    for idx, example in enumerate(examples, start=1):
        prompt = format_prompt(example)
        prompt_ids, _ = encode_prompt(tokenizer, start_token.token_id, prompt, None, max_original_len)
        b0, b1, d1 = relocation_metrics(model, prompt_ids, bias_by_layer, scale)
        bos_before += torch.tensor(b0, dtype=torch.float64)
        bos_after += torch.tensor(b1, dtype=torch.float64)
        dummy_after += torch.tensor(d1, dtype=torch.float64)
        total_forwards += 2

        clean_scores: list[float] = []
        changed_scores: list[float] = []
        for label in CHOICE_LABELS:
            input_ids, answer_start = encode_prompt(tokenizer, start_token.token_id, prompt, label, max_original_len)
            loss, count = score_choice(
                model,
                input_ids,
                answer_start,
                bias_by_layer=None,
                scale=scale,
                token_chunk=token_chunk,
            )
            clean_loss += loss
            clean_tokens += count
            clean_scores.append(-loss / max(1, count))
            total_forwards += 1
            if intervened_accuracy:
                iloss, icount = score_choice(
                    model,
                    input_ids,
                    answer_start,
                    bias_by_layer=bias_by_layer,
                    scale=scale,
                    token_chunk=token_chunk,
                )
                changed_loss += iloss
                changed_tokens += icount
                changed_scores.append(-iloss / max(1, icount))
                total_forwards += 1

        pred = int(torch.tensor(clean_scores).argmax().item())
        clean_correct += int(pred == int(example["answer"]))
        if intervened_accuracy:
            ipred = int(torch.tensor(changed_scores).argmax().item())
            changed_correct += int(ipred == int(example["answer"]))
        if idx % 25 == 0:
            print(f"  seen {idx}/{len(examples)} examples", flush=True)

    denom = float(len(examples))
    bos_before /= denom
    bos_after /= denom
    dummy_after /= denom
    out = {
        "clean_accuracy": clean_correct / denom,
        "intervened_accuracy": changed_correct / denom if intervened_accuracy else float("nan"),
        "accuracy_delta": (changed_correct - clean_correct) / denom if intervened_accuracy else float("nan"),
        "clean_mean_choice_nll": clean_loss / max(1, clean_tokens),
        "intervened_mean_choice_nll": changed_loss / max(1, changed_tokens) if intervened_accuracy else float("nan"),
        "bos_before_mean": float(bos_before.float().mean().item()),
        "bos_after_mean": float(bos_after.float().mean().item()),
        "dummy_after_mean": float(dummy_after.float().mean().item()),
        "dummy_minus_bos_mean": float((dummy_after - bos_after).float().mean().item()),
        "total_forwards": float(total_forwards),
    }
    return out


def run_one(args: argparse.Namespace, row: dict[str, str], examples: list[dict[str, Any]]) -> dict[str, Any]:
    model_key = row["model_key"]
    model_id = row["model_id"]
    mask = row["mask"]
    if mask != "bos_only":
        raise ValueError(f"MMLU runner expects bos_only rows, got {mask}")
    out_csv = args.out / model_key / "mmlu_result.csv"
    if out_csv.exists() and not args.force:
        return next(csv.DictReader(out_csv.open()))

    selected = load_selected(model_key, row)
    window_length = window_length_for_model(model_id)
    train_windows = load_train_windows(model_key, window_length)
    train_mask = make_masks(train_windows, torch.zeros_like(train_windows, dtype=torch.bool))[mask]

    t_model = time.perf_counter()
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, eager_attention=True)
    model_load_seconds = time.perf_counter() - t_model
    max_original_len = model_context_limit(model, args.max_seq_len)

    old_batch = sink_mech.BATCH_SIZE
    sink_mech.BATCH_SIZE = args.bias_batch_size
    t_bias = time.perf_counter()
    try:
        bias_by_layer = estimate_bias(model, train_windows, selected, train_mask)
    finally:
        sink_mech.BATCH_SIZE = old_batch
    bias_seconds = time.perf_counter() - t_bias

    t_eval = time.perf_counter()
    metrics = evaluate_mmlu(
        model,
        tokenizer,
        examples,
        bias_by_layer,
        float(row["scale"]),
        max_original_len,
        args.loss_token_chunk,
        intervened_accuracy=not args.skip_intervened_accuracy,
    )
    eval_seconds = time.perf_counter() - t_eval

    subjects = sorted({example["subject"] for example in examples})
    result = {
        "model_key": model_key,
        "model_id": model_id,
        "status": "ok",
        "eval_dataset": args.dataset,
        "eval_config": args.config,
        "eval_split": args.split,
        "sample_mode": args.sample_mode,
        "eval_examples": len(examples),
        "subjects": " ".join(subjects),
        "mask": mask,
        "topk": int(row["topk"]),
        "scale": float(row["scale"]),
        "candidate_group": row.get("candidate_group", ""),
        "layers": layer_counts(selected),
        "bias_batch_size": args.bias_batch_size,
        "source_wikitext_dummy_minus_bos": row.get("dummy_minus_bos_mean"),
        "source_wikitext_ppl_ratio": row.get("ppl_ratio"),
        "model_load_seconds": model_load_seconds,
        "bias_seconds": bias_seconds,
        "eval_seconds": eval_seconds,
        "examples_per_second": len(examples) / eval_seconds if eval_seconds > 0 else float("nan"),
        "forwards_per_second": metrics["total_forwards"] / eval_seconds if eval_seconds > 0 else float("nan"),
        "max_original_seq_len": max_original_len,
        **metrics,
    }
    write_csv(out_csv, [result])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    configure_runtime()
    examples = load_mmlu_examples(args)
    rows_by_model = read_rows(args.rows_csv)
    results: list[dict[str, Any]] = []
    for model_key in args.models:
        print(f"[start] {model_key}", flush=True)
        try:
            result = run_one(args, rows_by_model[model_key], examples)
            print(
                f"[done] {model_key}: clean_acc={float(result['clean_accuracy']):.4f} "
                f"int_acc={float(result['intervened_accuracy']):.4f} "
                f"reloc={float(result['dummy_minus_bos_mean']):.6g} "
                f"eval_s={float(result['eval_seconds']):.2f}",
                flush=True,
            )
        except Exception as exc:
            result = {"model_key": model_key, "status": "error", "error": repr(exc)}
            print(f"[error] {model_key}: {exc!r}", flush=True)
        results.append(result)
        write_csv(args.out / "summary.csv", results)


if __name__ == "__main__":
    main()
