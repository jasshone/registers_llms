from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .artifacts import save_artifact
from .modeling import get_mlp_intermediate_size, get_transformer_layers
from .types import ScoreArtifactMetadata, normalize_path
from .windows import build_window_dataloader


@dataclass(frozen=True)
class BosScoreResult:
    scores: torch.Tensor
    num_examples: int
    z_scores: torch.Tensor
    bos_minus_nonbos: torch.Tensor
    nonbos_std: torch.Tensor


def _make_score_hook(
    layer_idx: int,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    bos_sum: torch.Tensor,
    nonbos_sum: torch.Tensor,
    nonbos_square_sum: torch.Tensor,
) -> Callable:
    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        gate_act = act_fn(output).float()
        nonbos = gate_act[:, 1:, :]
        bos_sum[layer_idx] += gate_act[:, 0, :].sum(dim=0).cpu()
        nonbos_sum[layer_idx] += nonbos.sum(dim=(0, 1)).cpu()
        nonbos_square_sum[layer_idx] += nonbos.square().sum(dim=(0, 1)).cpu()

    return hook


def _score_projection_and_activation(mlp: torch.nn.Module) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "act_fn"):
        return mlp.gate_proj, mlp.act_fn
    if hasattr(mlp, "c_fc") and hasattr(mlp, "act"):
        return mlp.c_fc, mlp.act
    if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "act"):
        return mlp.dense_h_to_4h, mlp.act
    if hasattr(mlp, "fc1") and hasattr(mlp, "activation_fn"):
        return mlp.fc1, mlp.activation_fn
    raise AttributeError("Unsupported MLP architecture for BOS scoring")


@torch.no_grad()
def score_bos_neurons(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
) -> BosScoreResult:
    num_layers = model.config.num_hidden_layers
    intermediate_size = get_mlp_intermediate_size(model)
    bos_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    nonbos_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    nonbos_square_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    total_examples = 0
    total_nonbos_positions = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layers = get_transformer_layers(model)

    for layer_idx in range(num_layers):
        projection, act_fn = _score_projection_and_activation(layers[layer_idx].mlp)
        handle = projection.register_forward_hook(_make_score_hook(layer_idx, act_fn, bos_sum, nonbos_sum, nonbos_square_sum))
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
    nonbos_mean_square = (nonbos_square_sum / total_nonbos_positions).float()
    nonbos_var = (nonbos_mean_square - nonbos_mean.square()).clamp_min(0.0)
    nonbos_std = torch.sqrt(nonbos_var).clamp_min(1e-8)
    bos_minus_nonbos = scores - nonbos_mean
    z_scores = bos_minus_nonbos / nonbos_std
    return BosScoreResult(
        scores=scores,
        num_examples=total_examples,
        z_scores=z_scores,
        bos_minus_nonbos=bos_minus_nonbos,
        nonbos_std=nonbos_std,
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
            "nonbos_std": result.nonbos_std,
        },
        metadata=metadata,
        config={"batch_size": batch_size},
    )
