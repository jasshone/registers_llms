from __future__ import annotations

import argparse
import csv
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

_RUNTIME_READY = False


def _ensure_runtime_imports() -> None:
    global _RUNTIME_READY
    global sink_mech, find_attention_sink_tokens
    global BATCH_SIZE, clean_baseline, emergence_layers, estimate_bias, layer_counts, make_masks
    global patched_bias_intervention, score_candidate_neurons, window_nll
    global load_pt, save_json, save_pt, configure_runtime, build_intervened_inputs
    global get_transformer_layers, load_model, load_tokenizer, measure_relocation_baseline
    global build_window_dataloader, build_windows_tensor, load_windows_artifact
    if _RUNTIME_READY:
        return
    import run_autoresearch_sink_mechanism as sink_mech_module
    from find_register_neurons import find_attention_sink_tokens as imported_find_attention_sink_tokens
    from run_autoresearch_sink_mechanism import (
        BATCH_SIZE as imported_batch_size,
        clean_baseline as imported_clean_baseline,
        emergence_layers as imported_emergence_layers,
        estimate_bias as imported_estimate_bias,
        layer_counts as imported_layer_counts,
        make_masks as imported_make_masks,
        patched_bias_intervention as imported_patched_bias_intervention,
        score_candidate_neurons as imported_score_candidate_neurons,
        window_nll as imported_window_nll,
    )
    from sink_neurons.artifacts import load_pt as imported_load_pt, save_json as imported_save_json, save_pt as imported_save_pt
    from sink_neurons.env import configure_runtime as imported_configure_runtime
    from sink_neurons.intervention import build_intervened_inputs as imported_build_intervened_inputs
    from sink_neurons.modeling import get_transformer_layers as imported_get_transformer_layers, load_model as imported_load_model, load_tokenizer as imported_load_tokenizer
    from sink_neurons.relocation import measure_relocation_baseline as imported_measure_relocation_baseline
    from sink_neurons.windows import build_window_dataloader as imported_build_window_dataloader, build_windows_tensor as imported_build_windows_tensor, load_windows_artifact as imported_load_windows_artifact

    sink_mech = sink_mech_module
    find_attention_sink_tokens = imported_find_attention_sink_tokens
    BATCH_SIZE = imported_batch_size
    clean_baseline = imported_clean_baseline
    emergence_layers = imported_emergence_layers
    estimate_bias = imported_estimate_bias
    layer_counts = imported_layer_counts
    make_masks = imported_make_masks
    patched_bias_intervention = imported_patched_bias_intervention
    score_candidate_neurons = imported_score_candidate_neurons
    window_nll = imported_window_nll
    load_pt = imported_load_pt
    save_json = imported_save_json
    save_pt = imported_save_pt
    configure_runtime = imported_configure_runtime
    build_intervened_inputs = imported_build_intervened_inputs
    get_transformer_layers = imported_get_transformer_layers
    load_model = imported_load_model
    load_tokenizer = imported_load_tokenizer
    measure_relocation_baseline = imported_measure_relocation_baseline
    build_window_dataloader = imported_build_window_dataloader
    build_windows_tensor = imported_build_windows_tensor
    load_windows_artifact = imported_load_windows_artifact
    _RUNTIME_READY = True


DEFAULT_OUT = Path("outputs/final_split_bias_transfer")
WINDOWS_PER_SPLIT = 64
MASKS = ["bos_only", "pipeline_sinks_plus_bos"]
NORMAL_TOPKS = [1, 2, 4, 8, 16]
RESCUE_TOPKS = [1, 2, 4, 8, 16, 32]
PPL_LIMIT = 1.06
FAMILIES = ["gpt2", "pythia", "llama", "qwen3", "opt", "mistral", "phi"]

SCALE_PRESETS: dict[str, list[float]] = {
    "core": [0.9, 1.0, 1.25, 1.5, 2.0, 3.0],
    "rescue": [0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 7.0, 8.0, 10.0],
    "extended": [
        0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5,
        3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0,
        12.0, 15.0, 16.0, 20.0, 24.0, 30.0, 32.0,
    ],
}


@dataclass(frozen=True)
class ReproductionPreset:
    description: str
    models: tuple[str, ...]
    windows_per_split: int = WINDOWS_PER_SPLIT
    bias_windows: int = WINDOWS_PER_SPLIT
    scale_preset: str = "core"
    rescue: str = "failed"
    rescue_scale_preset: str = "rescue"
    topks: tuple[int, ...] = tuple(NORMAL_TOPKS)
    rescue_topks: tuple[int, ...] = tuple(RESCUE_TOPKS)


REPRODUCTION_PRESETS: dict[str, ReproductionPreset] = {
    "smoke-gpt2": ReproductionPreset(
        description="Fast verified GPT-2 smoke run; downloads gpt2 and writes a known-good result row.",
        models=("gpt2",),
        windows_per_split=4,
        bias_windows=4,
        rescue="off",
    ),
    "smoke-cross-family": ReproductionPreset(
        description="Cheap cross-family smoke across GPT-2 and Pythia-70M.",
        models=("gpt2", "pythia_70m"),
        windows_per_split=4,
        bias_windows=4,
        rescue="off",
    ),
    "smoke-scale-sweep-gpt2": ReproductionPreset(
        description="Cheap GPT-2 smoke that exercises the full standardized extended scale grid.",
        models=("gpt2",),
        windows_per_split=4,
        bias_windows=4,
        scale_preset="extended",
        rescue="off",
    ),
    "gpt2-family": ReproductionPreset(
        description="Core final-split reproduction for GPT-2, medium, large, and XL.",
        models=("gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    ),
    "small-open-models": ReproductionPreset(
        description="Small public checkpoints spanning GPT-2, Pythia, Qwen3, OPT, and Phi.",
        models=("gpt2", "pythia_70m", "pythia_160m", "qwen3_0_6b", "opt_125m", "opt_350m", "phi_1_5"),
    ),
    "qwen3-small": ReproductionPreset(
        description="Qwen3 small-to-mid models with the standardized core sweep.",
        models=("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b"),
    ),
    "pythia-core": ReproductionPreset(
        description="Pythia family core sweep through 2.8B.",
        models=("pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "pythia_1_4b", "pythia_2_8b"),
    ),
    "paper-core": ReproductionPreset(
        description="Broad public core table over commonly used families; larger models may require substantial GPU memory.",
        models=("gpt2", "gpt2_medium", "pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "qwen3_0_6b", "qwen3_1_7b", "opt_125m", "opt_350m", "phi_1_5"),
    ),
    "extended-scales-gpt2": ReproductionPreset(
        description="GPT-2 with the full standardized extended scale grid for scale-sweep reproduction.",
        models=("gpt2",),
        scale_preset="extended",
        rescue="off",
    ),
}


@dataclass(frozen=True)
class ModelCfg:
    family: str
    key: str
    model_id: str
    window_length: int
    existing_windows: tuple[str, ...] = ()
    eager_attention: bool = True


CONFIGS = [
    ModelCfg("gpt2", "gpt2", "gpt2", 1023, ("outputs/gpt2_confirmation/windows_128x1023.pt",)),
    ModelCfg("gpt2", "gpt2_medium", "gpt2-medium", 1023, ("outputs/gpt2_confirmation/windows_128x1023.pt",)),
    ModelCfg("gpt2", "gpt2_large", "gpt2-large", 1023),
    ModelCfg("gpt2", "gpt2_xl", "gpt2-xl", 1023),
    ModelCfg("pythia", "pythia_70m", "EleutherAI/pythia-70m", 1024),
    ModelCfg("pythia", "pythia_160m", "EleutherAI/pythia-160m", 1024),
    ModelCfg("pythia", "pythia_410m", "EleutherAI/pythia-410m", 1024),
    ModelCfg("pythia", "pythia_1b", "EleutherAI/pythia-1b", 1024, ("outputs/pythia_1b/windows_128x1024.pt",)),
    ModelCfg("pythia", "pythia_1_4b", "EleutherAI/pythia-1.4b", 1024),
    ModelCfg("pythia", "pythia_2_8b", "EleutherAI/pythia-2.8b", 1024),
    ModelCfg("pythia", "pythia_6_9b", "EleutherAI/pythia-6.9b", 1024),
    ModelCfg("pythia", "pythia_12b", "EleutherAI/pythia-12b", 1024),
    ModelCfg("llama", "llama3_2_1b", "unsloth/Llama-3.2-1B", 1024, ("outputs/unsloth_llama3_2_1b/selection_ablation_v2_32w/windows_32x1024.pt",)),
    ModelCfg("llama", "llama3_2_3b", "unsloth/Llama-3.2-3B", 1024, ("outputs/unsloth_llama3_2_3b/windows_128x1024.pt",)),
    ModelCfg("llama", "llama3_8b", "unsloth/llama-3-8b", 1024, ("outputs/llama3_8b/windows_128x1024.pt",)),
    ModelCfg("qwen3", "qwen3_0_6b", "Qwen/Qwen3-0.6B", 1024),
    ModelCfg("qwen3", "qwen3_1_7b", "Qwen/Qwen3-1.7B", 1024, ("outputs/qwen3_1_7b/windows_32x1024.pt",)),
    ModelCfg("qwen3", "qwen3_4b", "Qwen/Qwen3-4B", 1024),
    ModelCfg("qwen3", "qwen3_8b", "Qwen/Qwen3-8B", 1024),
    ModelCfg("qwen3", "qwen3_14b", "Qwen/Qwen3-14B", 1024),
    ModelCfg("qwen3", "qwen3_30b_a3b", "Qwen/Qwen3-30B-A3B", 1024),
    ModelCfg("opt", "opt_125m", "facebook/opt-125m", 1024),
    ModelCfg("opt", "opt_350m", "facebook/opt-350m", 1024),
    ModelCfg("opt", "opt_1_3b", "facebook/opt-1.3b", 1024),
    ModelCfg("opt", "opt_2_7b", "facebook/opt-2.7b", 1024),
    ModelCfg("opt", "opt_6_7b", "facebook/opt-6.7b", 1024),
    ModelCfg("opt", "opt_13b", "facebook/opt-13b", 1024),
    ModelCfg("mistral", "mistral_7b_v0_1", "mistralai/Mistral-7B-v0.1", 1024),
    ModelCfg("mistral", "mistral_7b_v0_3", "mistralai/Mistral-7B-v0.3", 1024),
    ModelCfg("mistral", "mistral_nemo_12b", "mistralai/Mistral-Nemo-Base-2407", 1024),
    ModelCfg("phi", "phi_1_5", "microsoft/phi-1_5", 1024),
    ModelCfg("phi", "phi_2", "microsoft/phi-2", 1024),
    ModelCfg("phi", "phi3_mini_4k", "microsoft/Phi-3-mini-4k-instruct", 1024),
    ModelCfg("phi", "phi3_medium_4k", "microsoft/Phi-3-medium-4k-instruct", 1024),
    ModelCfg("phi", "phi3_5_mini", "microsoft/Phi-3.5-mini-instruct", 1024),
    ModelCfg("phi", "phi4", "microsoft/phi-4", 1024),
]


MODEL_SETS: dict[str, tuple[str, ...]] = {
    "gpt2": ("gpt2",),
    "gpt2-family": ("gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl"),
    "gpt2-pythia": ("gpt2", "pythia_70m"),
    "pythia-core": ("pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "pythia_1_4b", "pythia_2_8b"),
    "qwen3-small": ("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b"),
    "small-open": ("gpt2", "pythia_70m", "pythia_160m", "qwen3_0_6b", "opt_125m", "opt_350m", "phi_1_5"),
    "paper-core": ("gpt2", "gpt2_medium", "pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "qwen3_0_6b", "qwen3_1_7b", "opt_125m", "opt_350m", "phi_1_5"),
}


def scale_preset(name: str) -> list[float]:
    try:
        return list(SCALE_PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"unknown scale preset {name!r}; choose from {sorted(SCALE_PRESETS)}") from exc


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "family", "model_key", "model_id", "split", "status", "mask", "mode", "candidate_group",
        "topk", "scale", "layers", "candidate_layers", "sink_positions", "bos_before_mean",
        "bos_after_mean", "dummy_after_mean", "dummy_minus_bos_mean", "baseline_ppl",
        "intervened_ppl", "ppl_ratio", "ppl_change_pct", "delta_mean_nll", "selection_reason",
        "intervention_applied", "intervention_method", "intervention_reason", "error",
    ]
    keys = []
    all_keys = sorted({key for row in rows for key in row})
    keys.extend([key for key in preferred if key in all_keys])
    keys.extend([key for key in all_keys if key not in keys])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_windows_from_any(paths: Iterable[str], need: int) -> torch.Tensor | None:
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        try:
            windows, _payload = load_windows_artifact(path)
        except Exception:
            payload = load_pt(path)
            windows = payload.get("windows")
            if windows is None:
                windows = payload.get("tensors", {}).get("windows")
        if isinstance(windows, torch.Tensor) and int(windows.shape[0]) >= need:
            return windows[:need].long().clone()
    return None


def get_or_build_windows(cfg: ModelCfg, out_dir: Path, total_windows: int) -> torch.Tensor:
    windows_path = out_dir / f"windows_{total_windows}x{cfg.window_length}.pt"
    if windows_path.exists():
        return load_pt(windows_path)["windows"].long()
    windows = load_windows_from_any(cfg.existing_windows, total_windows)
    if windows is None:
        tokenizer = load_tokenizer(cfg.model_id)
        windows, stream_len = build_windows_tensor(
            tokenizer,
            split="train",
            window_length=cfg.window_length,
            stride=cfg.window_length - 1,
            max_windows=total_windows,
            show_progress=True,
        )
        save_pt(windows_path, {"windows": windows, "tokenized_stream_length": stream_len, "model_id": cfg.model_id})
    else:
        save_pt(windows_path, {"windows": windows, "model_id": cfg.model_id, "source": "existing_or_cached"})
    return windows.long()


def split_windows(windows: torch.Tensor, n: int) -> dict[str, torch.Tensor]:
    required = 3 * n
    if int(windows.shape[0]) < required:
        raise ValueError(f"need {required} windows but only have {windows.shape[0]}")
    return {"train": windows[:n].clone(), "val": windows[n:2 * n].clone(), "test": windows[2 * n:3 * n].clone()}


def compute_or_load_attention_sink_mask(model: torch.nn.Module, windows: torch.Tensor, path: Path) -> torch.Tensor:
    if path.exists():
        payload = load_pt(path)
        cached = payload.get("sink_mask")
        if isinstance(cached, torch.Tensor) and tuple(cached.shape) == tuple(windows.shape):
            return cached.bool()
        print(f"[cache-miss] {path}: cached mask shape does not match {tuple(windows.shape)}", flush=True)
    sink_mask, max_received, hit_count, sink_counts = find_attention_sink_tokens(
        model,
        windows,
        batch_size=BATCH_SIZE,
        attention_threshold=0.2,
        min_query_count=32,
        exclude_positions=set(),
        show_progress=True,
    )
    save_pt(path, {
        "sink_mask": sink_mask.bool(),
        "max_attention_received": max_received,
        "sink_hit_count": hit_count,
        "sink_counts": sink_counts,
        "sink_source": "attention_sink",
    })
    return sink_mask.bool()


@torch.no_grad()
def find_massive_activation_tokens(
    model: torch.nn.Module,
    windows: torch.Tensor,
    *,
    batch_size: int,
    abs_threshold: float,
    median_ratio: float,
    min_future_queries: int,
    exclude_positions: set[int],
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sink_mask = torch.zeros(windows.shape, dtype=torch.bool)
    max_activation = torch.zeros(windows.shape, dtype=torch.float32)
    hit_count = torch.zeros(windows.shape, dtype=torch.long)
    sink_counts = torch.zeros(windows.shape[0], dtype=torch.long)
    candidate_mask = torch.ones(windows.shape, dtype=torch.bool)
    for position in sorted(exclude_positions):
        if 0 <= position < windows.shape[1]:
            candidate_mask[:, position] = False
    seq_len = int(windows.shape[1])
    valid_future = torch.arange(seq_len - 1, -1, -1) >= int(min_future_queries)
    iterator = range(0, windows.shape[0], batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Finding massive-activation source tokens", total=(windows.shape[0] + batch_size - 1) // batch_size, dynamic_ncols=True)
    for start in iterator:
        end = min(start + batch_size, windows.shape[0])
        batch = windows[start:end].to(model.device)
        outputs = model(input_ids=batch, output_hidden_states=True, use_cache=False, return_dict=True)
        batch_max = torch.zeros((end - start, seq_len), dtype=torch.float32)
        batch_hits = torch.zeros((end - start, seq_len), dtype=torch.long)
        for hidden in outputs.hidden_states[1:]:
            hidden_abs = hidden.detach().float().abs()
            flat = hidden_abs.reshape(hidden_abs.shape[0], -1)
            med = flat.median(dim=1).values.clamp_min(1e-12)
            threshold = torch.maximum(torch.full_like(med, float(abs_threshold)), med * float(median_ratio))
            token_max = hidden_abs.amax(dim=-1)
            passes = token_max >= threshold[:, None]
            batch_max = torch.maximum(batch_max, token_max.cpu())
            batch_hits += passes.cpu().long()
        batch_candidate_mask = candidate_mask[start:end] & valid_future[None, :]
        batch_sink_mask = batch_candidate_mask & (batch_hits > 0)
        sink_mask[start:end] = batch_sink_mask
        max_activation[start:end] = batch_max
        hit_count[start:end] = batch_hits * batch_candidate_mask.long()
        sink_counts[start:end] = batch_sink_mask.sum(dim=1)
    return sink_mask, max_activation, hit_count, sink_counts


def compute_or_load_source_mask(model: torch.nn.Module, windows: torch.Tensor, path: Path, args: argparse.Namespace) -> torch.Tensor:
    if path.exists():
        payload = load_pt(path)
        cached = payload.get("sink_mask")
        if isinstance(cached, torch.Tensor) and tuple(cached.shape) == tuple(windows.shape):
            return cached.bool()
        print(f"[cache-miss] {path}: cached mask shape does not match {tuple(windows.shape)}", flush=True)
    if args.sink_source == "attention_sink":
        return compute_or_load_attention_sink_mask(model, windows, path)
    sink_mask, max_activation, hit_count, sink_counts = find_massive_activation_tokens(
        model,
        windows,
        batch_size=BATCH_SIZE,
        abs_threshold=args.massive_abs_threshold,
        median_ratio=args.massive_median_ratio,
        min_future_queries=args.massive_min_future_queries,
        exclude_positions=set(args.massive_exclude_positions),
        show_progress=True,
    )
    save_pt(path, {
        "sink_mask": sink_mask.bool(),
        "max_massive_activation": max_activation,
        "massive_hit_count": hit_count,
        "sink_counts": sink_counts,
        "sink_source": args.sink_source,
        "abs_threshold": args.massive_abs_threshold,
        "median_ratio": args.massive_median_ratio,
        "min_future_queries": args.massive_min_future_queries,
        "exclude_positions": args.massive_exclude_positions,
    })
    return sink_mask.bool()


@torch.no_grad()
def evaluate_with_fixed_bias(
    model: torch.nn.Module,
    windows: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    mode: str,
    scale: float,
) -> dict[str, float]:
    if selected.numel() == 0:
        return {"dummy_minus_bos_mean": float("-inf"), "ppl_ratio": float("inf")}
    baseline = measure_relocation_baseline(model, windows, batch_size=BATCH_SIZE, show_progress=False)
    depth = int(baseline.bos_attention_before.numel())
    bos_after = torch.zeros(depth, dtype=torch.float64)
    dummy_after = torch.zeros(depth, dtype=torch.float64)
    clean_loss = 0.0
    changed_loss = 0.0
    tokens = 0
    examples = 0
    offset = 0
    for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
        bsz = int(batch.shape[0])
        src_batch = source_mask[offset: offset + bsz]
        offset += bsz
        batch = batch.to(model.device)
        clean = model(input_ids=batch, use_cache=False, return_dict=True)
        labels = batch.clone()
        labels[:, 0] = -100
        loss, count = window_nll(clean.logits, labels)
        clean_loss += float(loss.sum().item())
        tokens += int(count.sum().item())
        with patched_bias_intervention(model, bias_by_layer, src_batch, scale, mode):
            embeds, attention_mask = build_intervened_inputs(model, batch, dummy_init="zero", num_dummy_tokens=1)
            changed = model(inputs_embeds=embeds, attention_mask=attention_mask, output_attentions=True, use_cache=False, return_dict=True)
        ilabels = torch.full((batch.shape[0], batch.shape[1] + 1), -100, dtype=torch.long, device=batch.device)
        ilabels[:, 2:] = batch[:, 1:]
        iloss, icount = window_nll(changed.logits, ilabels)
        if int(icount.sum().item()) != int(count.sum().item()):
            raise RuntimeError("token mismatch")
        changed_loss += float(iloss.sum().item())
        examples += bsz
        for layer_idx, attn in enumerate(changed.attentions):
            bos_after[layer_idx] += attn[:, :, 2:, 0].float().mean(dim=(1, 2)).sum().item()
            dummy_after[layer_idx] += attn[:, :, 2:, 1].float().mean(dim=(1, 2)).sum().item()
    bos_after = (bos_after / examples).float()
    dummy_after = (dummy_after / examples).float()
    clean_nll = clean_loss / tokens
    changed_nll = changed_loss / tokens
    ppl_ratio = math.exp(changed_nll - clean_nll)
    return {
        "bos_before_mean": float(baseline.bos_attention_before.mean().item()),
        "bos_after_mean": float(bos_after.mean().item()),
        "dummy_after_mean": float(dummy_after.mean().item()),
        "dummy_minus_bos_mean": float((dummy_after - bos_after).mean().item()),
        "baseline_ppl": math.exp(clean_nll),
        "intervened_ppl": math.exp(changed_nll),
        "ppl_ratio": ppl_ratio,
        "ppl_change_pct": (ppl_ratio - 1.0) * 100.0,
        "delta_mean_nll": changed_nll - clean_nll,
    }


@torch.no_grad()
def score_candidates(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    layers: list[int],
    *,
    max_top: int,
    source_abs_bonus: float,
) -> tuple[torch.Tensor, list[tuple[float, int, int, float, float, float]]]:
    old_bonus = sink_mech.SOURCE_ABS_BONUS
    old_topks = list(sink_mech.TOPKS)
    sink_mech.SOURCE_ABS_BONUS = float(source_abs_bonus)
    sink_mech.TOPKS = [int(max_top)]
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="sink_candidate_scores_") as tmp:
            result = score_candidate_neurons(model, windows, source_mask, layers, Path(tmp))
            payload = load_pt(Path(tmp) / "candidate_scores.pt")
    finally:
        sink_mech.SOURCE_ABS_BONUS = old_bonus
        sink_mech.TOPKS = old_topks
    selected = result["selected_top"].long()
    score_rows = payload.get("score_rows", [])[:500]
    return selected[:max_top].clone(), score_rows


def choose_validation_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transfers = [row for row in rows if row.get("mode") == "transfer" and row.get("status") == "ok"]
    if not transfers:
        raise ValueError("no successful transfer validation rows")
    positive = [row for row in transfers if safe_float(row.get("dummy_minus_bos_mean")) > 0]
    pool = positive or transfers
    strong = [row for row in pool if safe_float(row.get("dummy_minus_bos_mean")) >= 0.30]
    if strong:
        best = min(strong, key=lambda row: (safe_float(row.get("ppl_ratio"), math.inf), int(row["topk"]), -safe_float(row.get("dummy_minus_bos_mean"))))
        best["selection_reason"] = "reloc>=0.30_min_val_ppl"
        return best
    if positive:
        best = min(positive, key=lambda row: (safe_float(row.get("ppl_ratio"), math.inf), int(row["topk"]), -safe_float(row.get("dummy_minus_bos_mean"))))
        best["selection_reason"] = "positive_reloc_min_val_ppl"
        return best
    best = max(transfers, key=lambda row: safe_float(row.get("dummy_minus_bos_mean"), -math.inf))
    raise ValueError(f"no positive validation relocation; best relocation={safe_float(best.get('dummy_minus_bos_mean')):.6g}, ppl_ratio={safe_float(best.get('ppl_ratio')):.6g}")


def choose_rescue_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    positive = [row for row in ok if safe_float(row.get("dummy_minus_bos_mean")) > 0]
    clean = [row for row in positive if safe_float(row.get("ppl_ratio"), math.inf) <= PPL_LIMIT]
    if clean:
        best = max(clean, key=lambda row: (safe_float(row.get("dummy_minus_bos_mean")), -safe_float(row.get("ppl_ratio")), -int(row["topk"])))
        best["selection_reason"] = "rescue_max_val_reloc_with_ppl<=1.06"
        return best
    if positive:
        best = min(positive, key=lambda row: (safe_float(row.get("ppl_ratio"), math.inf), -safe_float(row.get("dummy_minus_bos_mean"))))
        best["selection_reason"] = "rescue_positive_min_val_ppl_no_row_under_1.06"
        return best
    best = max(ok, key=lambda row: safe_float(row.get("dummy_minus_bos_mean"), -math.inf))
    raise ValueError(f"rescue found no positive validation relocation; best={safe_float(best.get('dummy_minus_bos_mean')):.6g}, ppl={safe_float(best.get('ppl_ratio')):.6g}")


def audit_failure_reason(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing test_result.csv"
    reasons = []
    if row.get("status") != "ok":
        reasons.append(f"status={row.get('status')}; {str(row.get('error', ''))[:160]}")
    if safe_float(row.get("dummy_minus_bos_mean"), -math.inf) <= 0:
        reasons.append("no positive final relocation")
    if safe_float(row.get("ppl_ratio"), 1.0) > PPL_LIMIT:
        reasons.append(f"PPL ratio {safe_float(row.get('ppl_ratio')):.6g} > {PPL_LIMIT:g}")
    return "; ".join(reasons)


def row_prefix(cfg: ModelCfg, split: str, status: str = "ok") -> dict[str, Any]:
    return {"family": cfg.family, "model_key": cfg.key, "model_id": cfg.model_id, "split": split, "status": status}


def layer_groups(depth: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}

    def add(name: str, start: int, stop_inclusive: int) -> None:
        layers = [idx for idx in range(start, stop_inclusive + 1) if 0 <= idx < depth]
        if layers:
            groups[name] = layers

    add("early_0_8", 0, 8)
    add("early_0_5", 0, 5)
    add("early_1_7", 1, 7)
    add("early_2_6", 2, 6)
    add("first_quarter", 0, max(0, depth // 4 - 1))
    add("all_0_15", 0, min(depth - 1, 15))
    add("mid_8_23", 8, min(depth - 1, 23))
    add("late_16_31", 16, min(depth - 1, 31))
    add("second_half", depth // 2, depth - 1)
    add("all_0_31", 0, min(depth - 1, 31))
    return groups


def write_model_summary(out_dir: Path, validation_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> None:
    lines = [f"# {test_row['model_key']} Final Split Bias Transfer", "", "## Test Selection", ""]
    lines += ["| mask | k | scale | layers | relocation | PPL ratio | PPL change |", "|---|---:|---:|---|---:|---:|---:|"]
    lines.append(f"| {test_row['mask']} | {test_row['topk']} | {float(test_row['scale']):g} | {test_row['layers']} | {safe_float(test_row.get('dummy_minus_bos_mean')):.6g} | {safe_float(test_row.get('ppl_ratio')):.6g} | {safe_float(test_row.get('ppl_change_pct')):+.2f}% |")
    ok_val = [row for row in validation_rows if row.get("status") == "ok" and row.get("mode") == "transfer"]
    lines += ["", "## Top Validation Transfer Rows", "", "| mask | k | scale | layers | relocation | PPL ratio |", "|---|---:|---:|---|---:|---:|"]
    for row in sorted(ok_val, key=lambda item: (safe_float(item.get("ppl_ratio")), -safe_float(item.get("dummy_minus_bos_mean"))))[:10]:
        lines.append(f"| {row['mask']} | {row['topk']} | {float(row['scale']):g} | {row['layers']} | {safe_float(row.get('dummy_minus_bos_mean')):.6g} | {safe_float(row.get('ppl_ratio')):.6g} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_rescue_summary(out_dir: Path, rows: list[dict[str, Any]], test_row: dict[str, Any] | None, reason: str) -> None:
    lines = [f"# {out_dir.name} Clean Transfer Rescue", ""]
    lines.append("Intervention note: original final split output failed the audit; reran broader clean train/val source-vs-control transfer.")
    lines.append(f"Intervention reason: {reason}")
    if test_row:
        lines += ["", "## Final Rescue Test", "", "| group | mask | k | scale | layers | relocation | PPL ratio | PPL change |", "|---|---|---:|---:|---|---:|---:|---:|"]
        lines.append(f"| {test_row['candidate_group']} | {test_row['mask']} | {test_row['topk']} | {float(test_row['scale']):g} | {test_row['layers']} | {safe_float(test_row.get('dummy_minus_bos_mean')):.6g} | {safe_float(test_row.get('ppl_ratio')):.6g} | {safe_float(test_row.get('ppl_change_pct')):+.2f}% |")
    lines += ["", "## Top Validation Rows", "", "| group | mask | k | scale | layers | relocation | PPL ratio |", "|---|---|---:|---:|---|---:|---:|"]
    ok_val = [row for row in rows if row.get("status") == "ok"]
    for row in sorted(ok_val, key=lambda item: (-safe_float(item.get("dummy_minus_bos_mean"), -math.inf), safe_float(item.get("ppl_ratio"), math.inf)))[:20]:
        lines.append(f"| {row['candidate_group']} | {row['mask']} | {row['topk']} | {float(row['scale']):g} | {row['layers']} | {safe_float(row.get('dummy_minus_bos_mean')):.6g} | {safe_float(row.get('ppl_ratio')):.6g} |")
    (out_dir / "rescue_summary.md").write_text("\n".join(lines) + "\n")


def run_one(cfg: ModelCfg, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    _ensure_runtime_imports()
    out_dir = root / cfg.key
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test_result.csv"
    if test_path.exists() and not args.force:
        rows = read_csv(test_path)
        if rows:
            print(f"[skip] {cfg.key}: existing {test_path}", flush=True)
            return dict(rows[0])

    print(f"[start] {cfg.family}/{cfg.key}: {cfg.model_id}", flush=True)
    windows_per_split = int(args.windows_per_split)
    requested_bias_windows = int(args.bias_windows)
    bias_windows = max(windows_per_split, requested_bias_windows)
    extra_bias_windows = bias_windows - windows_per_split
    total_windows = windows_per_split * 3 + extra_bias_windows
    windows = get_or_build_windows(cfg, out_dir, total_windows)
    splits = split_windows(windows[: windows_per_split * 3], windows_per_split)
    bias_windows_tensor = torch.cat([splits["train"], windows[windows_per_split * 3: total_windows].clone()], dim=0) if extra_bias_windows > 0 else splits["train"]
    save_json(out_dir / "splits.json", {
        "model_key": cfg.key,
        "model_id": cfg.model_id,
        "window_length": cfg.window_length,
        "windows_per_split": windows_per_split,
        "bias_windows": bias_windows,
        "reproduction_preset": getattr(args, "preset", None),
        "model_set": getattr(args, "model_set", None),
        "run_profile": getattr(args, "run_profile", None),
        "scale_preset": args.scale_preset,
        "scales": [float(x) for x in args.active_scales],
        "slices": {"train": [0, windows_per_split], "val": [windows_per_split, 2 * windows_per_split], "test": [2 * windows_per_split, 3 * windows_per_split], "extra_bias": [windows_per_split * 3, total_windows]},
    })

    model = load_model(cfg.model_id, eager_attention=cfg.eager_attention)
    clean_train = clean_baseline(model, splits["train"])
    candidate_layers = emergence_layers(clean_train["bos_attention"])
    save_json(out_dir / "context.json", {
        "family": cfg.family,
        "model_key": cfg.key,
        "model_id": cfg.model_id,
        "window_length": cfg.window_length,
        "windows_per_split": windows_per_split,
        "bias_windows": bias_windows,
        "sink_source": args.sink_source,
        "candidate_layers": candidate_layers,
        "train_mean_bos_attention": clean_train["mean_bos"],
        "train_max_layer": clean_train["max_layer"],
        "train_max_bos_attention": clean_train["max_bos"],
        "train_episodes": clean_train["episodes"],
        "selection_rule": f"candidate emergence band; score=source-control+{args.source_abs_bonus:g}*abs(source); validation selects mask,k,scale; test reports selected row",
        "reproduction_preset": getattr(args, "preset", None),
        "model_set": getattr(args, "model_set", None),
        "run_profile": getattr(args, "run_profile", None),
        "source_abs_bonus": float(args.source_abs_bonus),
    })

    split_masks: dict[str, dict[str, torch.Tensor]] = {}
    for split_name, split_tensor in splits.items():
        source_mask = compute_or_load_source_mask(model, split_tensor, out_dir / f"{split_name}_{args.sink_source}_mask.pt", args)
        split_masks[split_name] = make_masks(split_tensor, source_mask)
    bias_source_mask = compute_or_load_source_mask(model, bias_windows_tensor, out_dir / f"bias{bias_windows}_{args.sink_source}_mask.pt", args)
    bias_masks = make_masks(bias_windows_tensor, bias_source_mask)

    validation_rows: list[dict[str, Any]] = []
    selected_cache: dict[tuple[str, int], tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]] = {}
    for mask_name in MASKS:
        candidates, score_rows = score_candidates(model, splits["train"], split_masks["train"][mask_name], candidate_layers, max_top=max(args.active_topks), source_abs_bonus=float(args.source_abs_bonus))
        save_pt(out_dir / mask_name / "candidate_scores.pt", {"selected_top": candidates, "score_rows": score_rows, "layers": candidate_layers, "source_abs_bonus": float(args.source_abs_bonus)})
        for topk in args.active_topks:
            if topk > int(candidates.shape[0]):
                continue
            selected = candidates[:topk].clone()
            bias_by_layer = estimate_bias(model, bias_windows_tensor, selected, bias_masks[mask_name])
            selected_cache[(mask_name, topk)] = (selected, bias_by_layer)
            specs = [("add_only", 1.0), ("subtract_only", 1.0)] + [("transfer", scale) for scale in args.active_scales]
            for mode, scale in specs:
                row = row_prefix(cfg, "val") | {
                    "mask": mask_name,
                    "mode": mode,
                    "topk": topk,
                    "scale": scale,
                    "layers": layer_counts(selected),
                    "candidate_layers": " ".join(str(x) for x in candidate_layers),
                    "sink_positions": int(split_masks["val"][mask_name].sum().item()),
                    "source_abs_bonus": float(args.source_abs_bonus),
                }
                try:
                    row.update(evaluate_with_fixed_bias(model, splits["val"], selected, split_masks["val"][mask_name], bias_by_layer, mode, scale))
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = repr(exc)
                validation_rows.append(row)
                write_csv(out_dir / "validation_grid.csv", validation_rows)
                print(f"[val] {cfg.key} {mask_name} k={topk} {mode} s={scale:g}: reloc={safe_float(row.get('dummy_minus_bos_mean')):.4g} ppl={safe_float(row.get('ppl_ratio')):.4g} status={row['status']}", flush=True)

    best = choose_validation_row(validation_rows)
    mask_name = str(best["mask"])
    topk = int(best["topk"])
    selected, bias_by_layer = selected_cache[(mask_name, topk)]
    test_row = row_prefix(cfg, "test") | {
        "mask": mask_name,
        "mode": "transfer",
        "topk": topk,
        "scale": float(best["scale"]),
        "layers": layer_counts(selected),
        "candidate_layers": " ".join(str(x) for x in candidate_layers),
        "sink_positions": int(split_masks["test"][mask_name].sum().item()),
        "selection_reason": best.get("selection_reason", ""),
        "val_dummy_minus_bos_mean": best.get("dummy_minus_bos_mean"),
        "val_ppl_ratio": best.get("ppl_ratio"),
        "source_abs_bonus": float(args.source_abs_bonus),
    }
    test_row.update(evaluate_with_fixed_bias(model, splits["test"], selected, split_masks["test"][mask_name], bias_by_layer, "transfer", float(best["scale"])))
    write_csv(test_path, [test_row])
    write_model_summary(out_dir, validation_rows, test_row)
    print(f"[done] {cfg.key}: test reloc={test_row['dummy_minus_bos_mean']:.4g} ppl={test_row['ppl_ratio']:.4g}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_row


def run_rescue(cfg: ModelCfg, root: Path, args: argparse.Namespace, reason: str) -> dict[str, Any]:
    _ensure_runtime_imports()
    out_dir = root / cfg.key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rescue-start] {cfg.key}: {reason}", flush=True)
    windows_per_split = int(args.windows_per_split)
    windows = get_or_build_windows(cfg, out_dir, windows_per_split * 3)
    splits = split_windows(windows, windows_per_split)
    save_json(out_dir / "rescue_context.json", {
        "model_key": cfg.key,
        "model_id": cfg.model_id,
        "windows_per_split": windows_per_split,
        "intervention_applied": True,
        "intervention_reason": reason,
        "method": "broader train source-vs-control candidate scoring; validation selects transfer row; test runs once after selection",
        "source_abs_bonus": float(args.source_abs_bonus),
        "reproduction_preset": getattr(args, "preset", None),
        "model_set": getattr(args, "model_set", None),
        "run_profile": getattr(args, "run_profile", None),
        "scale_preset": args.rescue_scale_preset,
        "scales": [float(x) for x in args.active_rescue_scales],
    })

    model = load_model(cfg.model_id, eager_attention=cfg.eager_attention)
    depth = len(get_transformer_layers(model))
    groups = layer_groups(depth)
    if args.groups:
        wanted = set(args.groups)
        groups = {name: layers for name, layers in groups.items() if name in wanted}
        missing = sorted(wanted - set(groups))
        if missing:
            raise KeyError(f"unknown or empty group(s): {missing}")
    masks = [name for name in MASKS if not args.masks or name in set(args.masks)]
    if not masks:
        raise KeyError(f"unknown mask filter: {args.masks}")
    split_masks: dict[str, dict[str, torch.Tensor]] = {}
    for split_name, split_tensor in splits.items():
        source_mask = compute_or_load_source_mask(model, split_tensor, out_dir / f"{split_name}_{args.sink_source}_mask.pt", args)
        split_masks[split_name] = make_masks(split_tensor, source_mask)

    rows: list[dict[str, Any]] = []
    selected_cache: dict[tuple[str, str, int], tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]] = {}
    for group_name, layers in groups.items():
        for mask_name in masks:
            selected_top, score_rows = score_candidates(model, splits["train"], split_masks["train"][mask_name], layers, max_top=max(args.active_rescue_topks), source_abs_bonus=float(args.source_abs_bonus))
            save_json(out_dir / f"rescue_scores_{group_name}_{mask_name}.json", {
                "group": group_name,
                "mask": mask_name,
                "layers": layers,
                "score": f"source_minus_control_plus_abs_source_{float(args.source_abs_bonus):g}",
                "top_rows": score_rows[:200],
                "selected_top": selected_top.cpu().long().tolist(),
            })
            for topk in args.active_rescue_topks:
                if topk > int(selected_top.shape[0]):
                    continue
                selected = selected_top[:topk].cpu().long().clone()
                bias_by_layer = estimate_bias(model, splits["train"], selected, split_masks["train"][mask_name])
                selected_cache[(group_name, mask_name, topk)] = (selected, bias_by_layer)
                for scale in args.active_rescue_scales:
                    row = row_prefix(cfg, "val") | {
                        "candidate_group": group_name,
                        "mask": mask_name,
                        "mode": "transfer",
                        "topk": topk,
                        "scale": scale,
                        "layers": layer_counts(selected),
                        "candidate_layers": " ".join(str(x) for x in layers),
                        "sink_positions": int(split_masks["val"][mask_name].sum().item()),
                        "intervention_applied": True,
                        "intervention_reason": reason,
                        "intervention_method": "clean_source_control_transfer_rescue",
                    }
                    try:
                        row.update(evaluate_with_fixed_bias(model, splits["val"], selected, split_masks["val"][mask_name], bias_by_layer, "transfer", scale))
                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = repr(exc)
                    rows.append(row)
                    write_csv(out_dir / "rescue_validation_grid.csv", rows)
                    write_rescue_summary(out_dir, rows, None, reason)
                    print(f"[rescue-val] {cfg.key} {group_name} {mask_name} k={topk} s={scale:g}: reloc={safe_float(row.get('dummy_minus_bos_mean')):.4g} ppl={safe_float(row.get('ppl_ratio')):.4g} status={row['status']}", flush=True)

    best = choose_rescue_row(rows)
    group_name = str(best["candidate_group"])
    mask_name = str(best["mask"])
    topk = int(best["topk"])
    scale = float(best["scale"])
    selected, bias_by_layer = selected_cache[(group_name, mask_name, topk)]
    test_row = row_prefix(cfg, "test") | {
        "candidate_group": group_name,
        "mask": mask_name,
        "mode": "transfer",
        "topk": topk,
        "scale": scale,
        "layers": layer_counts(selected),
        "candidate_layers": best["candidate_layers"],
        "sink_positions": int(split_masks["test"][mask_name].sum().item()),
        "selection_reason": best.get("selection_reason", ""),
        "val_dummy_minus_bos_mean": best.get("dummy_minus_bos_mean"),
        "val_ppl_ratio": best.get("ppl_ratio"),
        "intervention_applied": True,
        "intervention_reason": reason,
        "intervention_method": "clean_source_control_transfer_rescue",
    }
    test_row.update(evaluate_with_fixed_bias(model, splits["test"], selected, split_masks["test"][mask_name], bias_by_layer, "transfer", scale))
    write_csv(out_dir / "rescue_test_result.csv", [test_row])
    write_rescue_summary(out_dir, rows, test_row, reason)
    print(f"[rescue-done] {cfg.key}: test reloc={test_row['dummy_minus_bos_mean']:.4g} ppl={test_row['ppl_ratio']:.4g}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_row


def error_row(cfg: ModelCfg, split: str, exc: BaseException) -> dict[str, Any]:
    return row_prefix(cfg, split, "error") | {"error": repr(exc), "traceback": traceback.format_exc(limit=20)}


def format_family_table(family: str, rows: list[dict[str, Any]]) -> str:
    title = f"{family.upper() if family != 'qwen3' else 'Qwen3'} Family"
    family_rows = [row for row in rows if row.get("family") == family]
    header = f"## {title}\n\n| model | mask | k | scale | layers | reloc | ppl ratio | ppl change | source |\n|---|---|---:|---:|---|---:|---:|---:|---|"
    lines = [header]
    order = {cfg.key: idx for idx, cfg in enumerate(CONFIGS) if cfg.family == family}
    for row in sorted(family_rows, key=lambda item: order.get(str(item.get("model_key")), 10_000)):
        if row.get("status") != "ok":
            err = str(row.get("error", "error")).replace("\n", " ")[:80]
            lines.append(f"| {row.get('model_key')} | SKIPPED |  |  |  |  |  |  | {err} |")
            continue
        source = "rescue" if row.get("intervention_applied") else "final"
        lines.append(f"| {row.get('model_key')} | {row.get('mask')} | {row.get('topk')} | {float(row.get('scale')):g} | {row.get('layers')} | {safe_float(row.get('dummy_minus_bos_mean')):.4f} | {safe_float(row.get('ppl_ratio')):.4f} | {safe_float(row.get('ppl_change_pct')):+.2f}% | {source} |")
    return "\n".join(lines) + "\n"


def write_family_outputs(root: Path, all_rows: list[dict[str, Any]]) -> None:
    write_csv(root / "final_results.csv", all_rows)
    combined = ["# Final Split Bias Transfer Tables", ""]
    for family in FAMILIES:
        family_rows = [row for row in all_rows if row.get("family") == family]
        write_csv(root / f"{family}_family.csv", family_rows)
        family_md = format_family_table(family, all_rows)
        (root / f"{family}_family.md").write_text(family_md)
        combined.append(family_md)
    (root / "final_results.md").write_text("\n".join(combined) + "\n")


def selected_configs(args: argparse.Namespace) -> list[ModelCfg]:
    configs = CONFIGS
    if args.models:
        wanted = set(args.models)
        return [cfg for cfg in configs if cfg.key in wanted]
    if args.families:
        return [cfg for cfg in configs if cfg.family in set(args.families)]
    return configs


def run_pipeline(args: argparse.Namespace) -> list[dict[str, Any]]:
    _ensure_runtime_imports()
    configure_runtime()
    sink_mech.SOURCE_ABS_BONUS = float(args.source_abs_bonus)
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for cfg in selected_configs(args):
        try:
            row = run_one(cfg, root, args)
        except Exception as exc:
            row = error_row(cfg, "test", exc)
            model_dir = root / cfg.key
            model_dir.mkdir(parents=True, exist_ok=True)
            write_csv(model_dir / "test_result.csv", [row])
            (model_dir / "error.log").write_text(row.get("traceback", "") + "\n")
            print(f"[error] {cfg.key}: {repr(exc)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        reason = audit_failure_reason(row)
        should_rescue = args.rescue == "always" or (args.rescue == "failed" and bool(reason))
        if should_rescue:
            try:
                row = run_rescue(cfg, root, args, reason or "manual rescue requested")
            except Exception as exc:
                rescue_error = error_row(cfg, "test", exc)
                rescue_error["intervention_applied"] = True
                rescue_error["intervention_method"] = "clean_source_control_transfer_rescue"
                rescue_error["intervention_reason"] = reason or "manual rescue requested"
                write_csv(root / cfg.key / "rescue_test_result.csv", [rescue_error])
                print(f"[rescue-error] {cfg.key}: {repr(exc)}", flush=True)
        all_rows.append(row)
        write_family_outputs(root, all_rows)
    for cfg in CONFIGS:
        if any(row.get("model_key") == cfg.key for row in all_rows):
            continue
        rescue_path = root / cfg.key / "rescue_test_result.csv"
        normal_path = root / cfg.key / "test_result.csv"
        path = rescue_path if rescue_path.exists() else normal_path
        rows = read_csv(path)
        if rows:
            all_rows.append(dict(rows[0]))
    write_family_outputs(root, all_rows)
    return all_rows
