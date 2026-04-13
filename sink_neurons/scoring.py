from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .artifacts import save_artifact
from .types import ScoreArtifactMetadata, normalize_path
from .windows import build_window_dataloader


@dataclass(frozen=True)
class BosScoreResult:
    scores: torch.Tensor
    num_examples: int
    z_scores: torch.Tensor
    bos_minus_nonbos: torch.Tensor


def _make_score_hook(
    model: PreTrainedModel,
    layer_idx: int,
    bos_sum: torch.Tensor,
    nonbos_sum: torch.Tensor,
) -> Callable:
    layer = model.model.layers[layer_idx]

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        x = inputs[0]
        gate_act = layer.mlp.act_fn(layer.mlp.gate_proj(x)).float()
        bos_sum[layer_idx] += gate_act[:, 0, :].sum(dim=0).cpu()
        nonbos_sum[layer_idx] += gate_act[:, 1:, :].sum(dim=(0, 1)).cpu()

    return hook


@torch.no_grad()
def score_bos_neurons(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
) -> BosScoreResult:
    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    bos_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    nonbos_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    total_examples = 0
    total_nonbos_positions = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx in range(num_layers):
        handle = model.model.layers[layer_idx].mlp.register_forward_hook(
            _make_score_hook(model, layer_idx, bos_sum, nonbos_sum)
        )
        handles.append(handle)

    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    try:
        for batch in tqdm(dataloader, desc="Scoring BOS neurons", disable=not show_progress, dynamic_ncols=True):
            batch = batch.to(model.device)
            total_examples += int(batch.shape[0])
            total_nonbos_positions += int(batch.shape[0] * (batch.shape[1] - 1))
            _ = model(input_ids=batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    scores = (bos_sum / total_examples).float()
    nonbos_mean = (nonbos_sum / total_nonbos_positions).float()
    bos_minus_nonbos = scores - nonbos_mean
    z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-8)
    return BosScoreResult(
        scores=scores,
        num_examples=total_examples,
        z_scores=z_scores,
        bos_minus_nonbos=bos_minus_nonbos,
    )


def save_bos_scores_artifact(
    output_path: Path,
    *,
    result: BosScoreResult,
    model_id: str,
    window_artifact: Path,
    batch_size: int,
) -> None:
    metadata = ScoreArtifactMetadata(
        model_id=model_id,
        score_name="mean_bos_gate_activation",
        window_artifact=normalize_path(window_artifact),
        num_layers=int(result.scores.shape[0]),
        intermediate_size=int(result.scores.shape[1]),
        num_examples=result.num_examples,
    )
    save_artifact(
        output_path,
        tensors={
            "scores": result.scores,
            "z_scores": result.z_scores,
            "bos_minus_nonbos": result.bos_minus_nonbos,
        },
        metadata=metadata,
        config={"batch_size": batch_size},
    )
