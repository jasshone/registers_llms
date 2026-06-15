from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else ()
from transformers import PreTrainedModel

from .artifacts import save_artifact
from .intervention import (
    AssignmentStrategy,
    DummyInit,
    DummyPosition,
    build_intervened_inputs,
    InterventionBackend,
    patched_mlp_reroute_forwards,
)
from .types import RelocationArtifactMetadata, normalize_path
from .windows import build_window_dataloader
from .modeling import get_transformer_layers


def _num_transformer_layers(model: PreTrainedModel) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "num_hidden_layers"):
        return int(model.config.text_config.num_hidden_layers)
    return len(get_transformer_layers(model))


@dataclass(frozen=True)
class RelocationResult:
    bos_norm_before: torch.Tensor
    bos_norm_after: torch.Tensor
    dummy_norm_after: torch.Tensor
    dummy_norm_after_by_slot: torch.Tensor
    bos_attention_before: torch.Tensor
    bos_attention_after: torch.Tensor
    dummy_attention_after: torch.Tensor
    dummy_attention_after_by_slot: torch.Tensor
    summary_metric: torch.Tensor
    num_examples: int


@dataclass(frozen=True)
class RelocationBaseline:
    bos_norm_before: torch.Tensor
    bos_attention_before: torch.Tensor
    num_examples: int


@torch.no_grad()
def measure_relocation_baseline(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
) -> RelocationBaseline:
    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    num_layers = _num_transformer_layers(model)
    bos_norm_before = torch.zeros(num_layers, dtype=torch.float64)
    bos_attention_before = torch.zeros(num_layers, dtype=torch.float64)
    total_examples = 0

    for batch in tqdm(dataloader, desc="Relocation baseline batches", disable=not show_progress, dynamic_ncols=True):
        batch = batch.to(model.device)
        baseline = model(
            input_ids=batch,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        bsz = int(batch.shape[0])
        total_examples += bsz

        for layer_idx, hidden in enumerate(baseline.hidden_states[1:]):
            bos_norm_before[layer_idx] += torch.linalg.vector_norm(hidden[:, 0, :].float(), dim=-1).sum().item()
        for layer_idx, attn in enumerate(baseline.attentions):
            bos_attention_before[layer_idx] += attn[:, :, 1:, 0].float().mean(dim=(1, 2)).sum().item()

    return RelocationBaseline(
        bos_norm_before=(bos_norm_before / total_examples).float(),
        bos_attention_before=(bos_attention_before / total_examples).float(),
        num_examples=total_examples,
    )


@torch.no_grad()
def run_relocation_experiment(
    model: PreTrainedModel,
    windows: torch.Tensor,
    selected: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
    baseline: RelocationBaseline | None = None,
    dummy_init: DummyInit = "zero",
    num_dummy_tokens: int = 1,
    relocation_scale: float = 1.0,
    relocation_fraction: float = 1.0,
    assignment_strategy: AssignmentStrategy = "round_robin",
    selected_scores: torch.Tensor | None = None,
    dummy_position: DummyPosition = "after_bos",
    intervention_backend: InterventionBackend = "mlp",
) -> RelocationResult:
    if num_dummy_tokens < 1:
        raise ValueError("num_dummy_tokens must be at least 1")
    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    num_layers = _num_transformer_layers(model)
    bos_norm_after = torch.zeros(num_layers, dtype=torch.float64)
    dummy_norm_after_by_slot = torch.zeros((num_layers, num_dummy_tokens), dtype=torch.float64)
    bos_attention_after = torch.zeros(num_layers, dtype=torch.float64)
    dummy_attention_after_by_slot = torch.zeros((num_layers, num_dummy_tokens), dtype=torch.float64)
    total_examples = 0
    if baseline is None:
        baseline = measure_relocation_baseline(
            model,
            windows,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    for batch in tqdm(dataloader, desc="Relocation intervened batches", disable=not show_progress, dynamic_ncols=True):
        batch = batch.to(model.device)
        with patched_mlp_reroute_forwards(
            model,
            selected,
            num_dummy_tokens=num_dummy_tokens,
            relocation_scale=relocation_scale,
            relocation_fraction=relocation_fraction,
            assignment_strategy=assignment_strategy,
            selected_scores=selected_scores,
            dummy_position=dummy_position,
            intervention_backend=intervention_backend,
        ):
            inputs_embeds, attention_mask = build_intervened_inputs(
                model,
                batch,
                dummy_init=dummy_init,
                num_dummy_tokens=num_dummy_tokens,
                dummy_position=dummy_position,
            )
            intervened = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        bsz = int(batch.shape[0])
        total_examples += bsz
        if dummy_position == "after_bos":
            bos_idx = 0
            dummy_start = 1
            query_start = 1 + num_dummy_tokens
        elif dummy_position == "before_bos":
            dummy_start = 0
            bos_idx = num_dummy_tokens
            query_start = 1 + num_dummy_tokens
        else:
            raise ValueError(f"Unsupported dummy_position: {dummy_position}")

        for layer_idx, hidden in enumerate(intervened.hidden_states[1:]):
            bos_norm_after[layer_idx] += torch.linalg.vector_norm(hidden[:, bos_idx, :].float(), dim=-1).sum().item()
            dummy_norm_after_by_slot[layer_idx] += (
                torch.linalg.vector_norm(
                    hidden[:, dummy_start : dummy_start + num_dummy_tokens, :].float(),
                    dim=-1,
                )
                .sum(dim=0)
                .cpu()
            )

        for layer_idx, attn in enumerate(intervened.attentions):
            bos_attention_after[layer_idx] += attn[:, :, query_start:, bos_idx].float().mean(dim=(1, 2)).sum().item()
            dummy_attention_after_by_slot[layer_idx] += (
                attn[:, :, query_start:, dummy_start : dummy_start + num_dummy_tokens]
                .float()
                .mean(dim=(1, 2))
                .sum(dim=0)
                .cpu()
            )

    bos_norm_after = (bos_norm_after / total_examples).float()
    dummy_norm_after_by_slot = (dummy_norm_after_by_slot / total_examples).float()
    dummy_norm_after = dummy_norm_after_by_slot.sum(dim=1)
    bos_attention_after = (bos_attention_after / total_examples).float()
    dummy_attention_after_by_slot = (dummy_attention_after_by_slot / total_examples).float()
    dummy_attention_after = dummy_attention_after_by_slot.sum(dim=1)
    summary_metric = dummy_attention_after - bos_attention_after
    return RelocationResult(
        bos_norm_before=baseline.bos_norm_before,
        bos_norm_after=bos_norm_after,
        dummy_norm_after=dummy_norm_after,
        dummy_norm_after_by_slot=dummy_norm_after_by_slot,
        bos_attention_before=baseline.bos_attention_before,
        bos_attention_after=bos_attention_after,
        dummy_attention_after=dummy_attention_after,
        dummy_attention_after_by_slot=dummy_attention_after_by_slot,
        summary_metric=summary_metric,
        num_examples=total_examples,
    )


def save_relocation_artifact(
    output_path: Path,
    *,
    result: RelocationResult,
    model_id: str,
    window_artifact: Path,
    selection_artifact: Path,
    batch_size: int,
    dummy_init: DummyInit = "zero",
    num_dummy_tokens: int = 1,
    relocation_scale: float = 1.0,
    relocation_fraction: float = 1.0,
    assignment_strategy: AssignmentStrategy = "round_robin",
    dummy_position: DummyPosition = "after_bos",
    intervention_backend: InterventionBackend = "mlp",
) -> None:
    metadata = RelocationArtifactMetadata(
        model_id=model_id,
        window_artifact=normalize_path(window_artifact),
        selection_artifact=normalize_path(selection_artifact),
        num_layers=int(result.bos_norm_before.shape[0]),
        num_examples=result.num_examples,
        summary_metric_name="sum_dummy_attention_after_minus_bos_attention_after",
    )
    save_artifact(
        output_path,
        tensors={
            "bos_norm_before": result.bos_norm_before,
            "bos_norm_after": result.bos_norm_after,
            "dummy_norm_after": result.dummy_norm_after,
            "dummy_norm_after_by_slot": result.dummy_norm_after_by_slot,
            "bos_attention_before": result.bos_attention_before,
            "bos_attention_after": result.bos_attention_after,
            "dummy_attention_after": result.dummy_attention_after,
            "dummy_attention_after_by_slot": result.dummy_attention_after_by_slot,
            "summary_metric_by_layer": result.summary_metric,
            "summary_metric_mean": result.summary_metric.mean(),
        },
        metadata=metadata,
        config={
            "batch_size": batch_size,
            "dummy_init": dummy_init,
            "num_dummy_tokens": num_dummy_tokens,
            "relocation_scale": relocation_scale,
            "relocation_fraction": relocation_fraction,
            "assignment_strategy": assignment_strategy,
            "dummy_position": dummy_position,
            "intervention_backend": intervention_backend,
        },
    )
