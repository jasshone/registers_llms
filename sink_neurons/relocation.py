from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .artifacts import save_artifact
from .intervention import build_intervened_inputs, register_mlp_reroute_hooks
from .types import RelocationArtifactMetadata, normalize_path
from .windows import build_window_dataloader


@dataclass(frozen=True)
class RelocationResult:
    bos_norm_before: torch.Tensor
    bos_norm_after: torch.Tensor
    dummy_norm_after: torch.Tensor
    bos_attention_before: torch.Tensor
    bos_attention_after: torch.Tensor
    dummy_attention_after: torch.Tensor
    summary_metric: torch.Tensor
    num_examples: int


@torch.no_grad()
def run_relocation_experiment(
    model: PreTrainedModel,
    windows: torch.Tensor,
    selected: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
) -> RelocationResult:
    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    num_layers = model.config.num_hidden_layers
    bos_norm_before = torch.zeros(num_layers, dtype=torch.float64)
    bos_norm_after = torch.zeros(num_layers, dtype=torch.float64)
    dummy_norm_after = torch.zeros(num_layers, dtype=torch.float64)
    bos_attention_before = torch.zeros(num_layers, dtype=torch.float64)
    bos_attention_after = torch.zeros(num_layers, dtype=torch.float64)
    dummy_attention_after = torch.zeros(num_layers, dtype=torch.float64)
    total_examples = 0

    for batch in tqdm(dataloader, desc="Relocation batches", disable=not show_progress, dynamic_ncols=True):
        batch = batch.to(model.device)
        baseline = model(
            input_ids=batch,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        handles = register_mlp_reroute_hooks(model, selected)
        try:
            inputs_embeds, attention_mask = build_intervened_inputs(model, batch)
            intervened = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        bsz = int(batch.shape[0])
        total_examples += bsz

        for layer_idx, hidden in enumerate(baseline.hidden_states[1:]):
            bos_norm_before[layer_idx] += torch.linalg.vector_norm(hidden[:, 0, :].float(), dim=-1).sum().item()
        for layer_idx, hidden in enumerate(intervened.hidden_states[1:]):
            bos_norm_after[layer_idx] += torch.linalg.vector_norm(hidden[:, 0, :].float(), dim=-1).sum().item()
            dummy_norm_after[layer_idx] += torch.linalg.vector_norm(hidden[:, 1, :].float(), dim=-1).sum().item()

        for layer_idx, attn in enumerate(baseline.attentions):
            bos_attention_before[layer_idx] += attn[:, :, 1:, 0].float().mean(dim=(1, 2)).sum().item()
        for layer_idx, attn in enumerate(intervened.attentions):
            bos_attention_after[layer_idx] += attn[:, :, 2:, 0].float().mean(dim=(1, 2)).sum().item()
            dummy_attention_after[layer_idx] += attn[:, :, 2:, 1].float().mean(dim=(1, 2)).sum().item()

    bos_norm_before = (bos_norm_before / total_examples).float()
    bos_norm_after = (bos_norm_after / total_examples).float()
    dummy_norm_after = (dummy_norm_after / total_examples).float()
    bos_attention_before = (bos_attention_before / total_examples).float()
    bos_attention_after = (bos_attention_after / total_examples).float()
    dummy_attention_after = (dummy_attention_after / total_examples).float()
    summary_metric = dummy_attention_after - bos_attention_after
    return RelocationResult(
        bos_norm_before=bos_norm_before,
        bos_norm_after=bos_norm_after,
        dummy_norm_after=dummy_norm_after,
        bos_attention_before=bos_attention_before,
        bos_attention_after=bos_attention_after,
        dummy_attention_after=dummy_attention_after,
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
) -> None:
    metadata = RelocationArtifactMetadata(
        model_id=model_id,
        window_artifact=normalize_path(window_artifact),
        selection_artifact=normalize_path(selection_artifact),
        num_layers=int(result.bos_norm_before.shape[0]),
        num_examples=result.num_examples,
        summary_metric_name="dummy_attention_after_minus_bos_attention_after",
    )
    save_artifact(
        output_path,
        tensors={
            "bos_norm_before": result.bos_norm_before,
            "bos_norm_after": result.bos_norm_after,
            "dummy_norm_after": result.dummy_norm_after,
            "bos_attention_before": result.bos_attention_before,
            "bos_attention_after": result.bos_attention_after,
            "dummy_attention_after": result.dummy_attention_after,
            "summary_metric_by_layer": result.summary_metric,
            "summary_metric_mean": result.summary_metric.mean(),
        },
        metadata=metadata,
        config={"batch_size": batch_size},
    )
