from __future__ import annotations

from pathlib import Path

import torch

from .artifacts import save_artifact
from .types import SelectionArtifactMetadata, normalize_path


def flatten_selected(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).to(dtype=torch.long)


def select_neurons_by_percentile(scores: torch.Tensor, percentile: float) -> tuple[torch.Tensor, float]:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    flat = scores.float().reshape(-1)
    quantile = percentile / 100.0
    threshold = torch.quantile(flat, quantile).item()
    mask = scores >= threshold
    return flatten_selected(mask), float(threshold)


def select_neurons_by_topk(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, float]:
    flat = scores.float().reshape(-1)
    if topk <= 0 or topk > flat.numel():
        raise ValueError("topk must be between 1 and the total number of neurons")
    values, indices = torch.topk(flat, k=topk, largest=True)
    layers = indices // scores.shape[1]
    neurons = indices % scores.shape[1]
    selected = torch.stack([layers, neurons], dim=1).to(dtype=torch.long)
    threshold = float(values[-1].item())
    return selected, threshold


def select_neurons_by_z_score(z_scores: torch.Tensor, z_threshold: float) -> tuple[torch.Tensor, float]:
    mask = z_scores.float() >= z_threshold
    selected = flatten_selected(mask)
    if selected.numel() == 0:
        raise ValueError("z_threshold selected no neurons")
    selected_z_threshold = z_scores.float()[mask].min().item()
    return selected, float(selected_z_threshold)


def save_selection_artifact(
    output_path: Path,
    *,
    selected: torch.Tensor,
    model_id: str,
    score_artifact: Path,
    method: str,
    percentile: float | None,
    topk: int | None,
    threshold: float,
    z_threshold: float | None = None,
) -> None:
    metadata = SelectionArtifactMetadata(
        model_id=model_id,
        score_artifact=normalize_path(score_artifact),
        method=method,
        count=int(selected.shape[0]),
        percentile=percentile,
        topk=topk,
        z_threshold=z_threshold,
        threshold=threshold,
    )
    save_artifact(
        output_path,
        tensors={"selected": selected},
        metadata=metadata,
        config={
            "method": method,
            "percentile": percentile,
            "topk": topk,
            "z_threshold": z_threshold,
            "threshold": threshold,
        },
    )
