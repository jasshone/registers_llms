from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from tqdm.auto import tqdm

from .modeling import get_mlp_intermediate_size, get_transformer_layers


AblationScoreMethod = Literal[
    "gate_mean",
    "layer_relative_gate",
    "contribution_magnitude",
    "contribution_contrast",
    "direction_aware",
]

ControlTokenMode = Literal["all_non_sink", "same_window_random", "late_non_sink"]


@dataclass(frozen=True)
class AblationScoreResult:
    scores: torch.Tensor
    method: str
    sink_counts_by_layer: torch.Tensor
    control_counts_by_layer: torch.Tensor
    layer_mean_abs_gate: torch.Tensor
    gate_sink_mean: torch.Tensor
    gate_control_mean: torch.Tensor
    contribution_sink_mean: torch.Tensor
    contribution_control_mean: torch.Tensor
    direction_alignment: torch.Tensor | None


def _score_projection_activation_and_up_down(
    mlp: torch.nn.Module,
) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor], torch.nn.Module | None, torch.nn.Module | None]:
    if all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        return mlp.gate_proj, mlp.act_fn, mlp.up_proj, mlp.down_proj
    if all(hasattr(mlp, attr) for attr in ("c_fc", "c_proj", "act")):
        return mlp.c_fc, mlp.act, None, mlp.c_proj
    if all(hasattr(mlp, attr) for attr in ("dense_h_to_4h", "dense_4h_to_h", "act")):
        return mlp.dense_h_to_4h, mlp.act, None, mlp.dense_4h_to_h
    if all(hasattr(mlp, attr) for attr in ("fc1", "fc2", "activation_fn")):
        return mlp.fc1, mlp.activation_fn, None, mlp.fc2
    raise AttributeError("Unsupported MLP architecture for ablation scoring")


def _down_column_norms(down_projection: torch.nn.Module | None, intermediate_size: int) -> torch.Tensor:
    if down_projection is None or not hasattr(down_projection, "weight"):
        return torch.ones(intermediate_size, dtype=torch.float32)
    weight = down_projection.weight.detach().float()
    if weight.shape[1] == intermediate_size:
        return torch.linalg.vector_norm(weight, dim=0).cpu().clamp_min(1e-8)
    if weight.shape[0] == intermediate_size:
        return torch.linalg.vector_norm(weight, dim=1).cpu().clamp_min(1e-8)
    raise ValueError("Could not infer down-projection column dimension")


def _down_columns(down_projection: torch.nn.Module, intermediate_size: int) -> torch.Tensor:
    weight = down_projection.weight.detach().float()
    if weight.shape[1] == intermediate_size:
        return weight.t().cpu()
    if weight.shape[0] == intermediate_size:
        return weight.cpu()
    raise ValueError("Could not infer down-projection column dimension")


def build_control_mask(
    sink_mask: torch.Tensor,
    *,
    mode: ControlTokenMode,
    exclude_positions: set[int],
    late_fraction: float = 0.5,
) -> torch.Tensor:
    if sink_mask.ndim != 2:
        raise ValueError("sink_mask must have shape [windows, seq_len]")
    control_mask = ~sink_mask.bool()
    for position in sorted(exclude_positions):
        if 0 <= position < sink_mask.shape[1]:
            control_mask[:, position] = False

    if mode == "all_non_sink":
        return control_mask
    if mode == "late_non_sink":
        start = int(sink_mask.shape[1] * (1.0 - late_fraction))
        late_mask = torch.zeros_like(control_mask)
        late_mask[:, start:] = True
        return control_mask & late_mask
    if mode == "same_window_random":
        sampled = torch.zeros_like(control_mask)
        generator = torch.Generator().manual_seed(0)
        sink_counts = sink_mask.sum(dim=1).to(dtype=torch.long)
        for row_idx, count in enumerate(sink_counts.tolist()):
            if count <= 0:
                continue
            candidates = torch.nonzero(control_mask[row_idx], as_tuple=False).squeeze(1)
            if candidates.numel() == 0:
                continue
            take = min(int(count), int(candidates.numel()))
            perm = torch.randperm(candidates.numel(), generator=generator)[:take]
            sampled[row_idx, candidates[perm]] = True
        return sampled
    raise ValueError(f"Unsupported control token mode: {mode}")


@contextmanager
def _registered_hooks(handles: list[torch.utils.hooks.RemovableHandle]):
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def score_neurons_for_ablation(
    model: torch.nn.Module,
    windows: torch.Tensor,
    sink_mask: torch.Tensor,
    *,
    batch_size: int,
    method: AblationScoreMethod,
    control_mode: ControlTokenMode = "all_non_sink",
    exclude_control_positions: set[int] | None = None,
    top_layer: int | None = None,
    show_progress: bool = True,
) -> AblationScoreResult:
    if windows.shape != sink_mask.shape:
        raise ValueError("windows and sink_mask must have the same [windows, seq_len] shape")
    if exclude_control_positions is None:
        exclude_control_positions = {0}

    num_layers = int(model.config.num_hidden_layers)
    if top_layer is None:
        top_layer = num_layers - 1
    if top_layer < 0 or top_layer >= num_layers:
        raise ValueError("top_layer is outside the model layer range")
    intermediate_size = get_mlp_intermediate_size(model)
    layers = get_transformer_layers(model)
    control_mask = build_control_mask(
        sink_mask,
        mode=control_mode,
        exclude_positions=exclude_control_positions,
    )

    gate_sink_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    gate_control_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    abs_gate_all_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    contribution_sink_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    contribution_control_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    sink_hidden_sum = torch.zeros((num_layers, model.config.hidden_size), dtype=torch.float64)
    control_hidden_sum = torch.zeros((num_layers, model.config.hidden_size), dtype=torch.float64)
    sink_counts = torch.zeros(num_layers, dtype=torch.long)
    control_counts = torch.zeros(num_layers, dtype=torch.long)
    all_counts = torch.zeros(num_layers, dtype=torch.long)
    down_norms = torch.zeros((num_layers, intermediate_size), dtype=torch.float32)

    active_sink_mask: torch.Tensor | None = None
    active_control_mask: torch.Tensor | None = None
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int, act_fn: Callable[[torch.Tensor], torch.Tensor], up_proj, down_proj):
        down_norms[layer_idx] = _down_column_norms(down_proj, intermediate_size)

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_sink_mask is None or active_control_mask is None:
                raise RuntimeError("Active masks were not set for current batch")
            x = inputs[0].float()
            gate = act_fn(output).float()
            if up_proj is not None:
                channel_value = gate * up_proj(inputs[0]).float()
            else:
                channel_value = gate
            contribution = channel_value.abs() * down_norms[layer_idx].to(gate.device)

            sink_selected = active_sink_mask
            control_selected = active_control_mask
            if sink_selected.any():
                gate_sink_sum[layer_idx] += gate[sink_selected].sum(dim=0).cpu().double()
                contribution_sink_sum[layer_idx] += contribution[sink_selected].sum(dim=0).cpu().double()
                sink_hidden_sum[layer_idx] += x[sink_selected].sum(dim=0).cpu().double()
                sink_counts[layer_idx] += int(sink_selected.sum().item())
            if control_selected.any():
                gate_control_sum[layer_idx] += gate[control_selected].sum(dim=0).cpu().double()
                contribution_control_sum[layer_idx] += contribution[control_selected].sum(dim=0).cpu().double()
                control_hidden_sum[layer_idx] += x[control_selected].sum(dim=0).cpu().double()
                control_counts[layer_idx] += int(control_selected.sum().item())
            abs_gate_all_sum[layer_idx] += gate.abs().sum(dim=(0, 1)).cpu().double()
            all_counts[layer_idx] += int(gate.shape[0] * gate.shape[1])

        return hook

    for layer_idx in range(top_layer + 1):
        projection, act_fn, up_proj, down_proj = _score_projection_activation_and_up_down(layers[layer_idx].mlp)
        handles.append(projection.register_forward_hook(make_hook(layer_idx, act_fn, up_proj, down_proj)))

    with _registered_hooks(handles):
        for start in tqdm(
            range(0, windows.shape[0], batch_size),
            desc=f"Scoring {method}",
            disable=not show_progress,
            dynamic_ncols=True,
        ):
            end = min(start + batch_size, windows.shape[0])
            active_sink_mask = sink_mask[start:end].to(model.device).bool()
            active_control_mask = control_mask[start:end].to(model.device).bool()
            batch = windows[start:end].to(model.device)
            _ = model(input_ids=batch, use_cache=False, return_dict=True)

    active_sink_mask = None
    active_control_mask = None

    sink_den = sink_counts.clamp_min(1).to(dtype=torch.float64)[:, None]
    control_den = control_counts.clamp_min(1).to(dtype=torch.float64)[:, None]
    all_den = all_counts.clamp_min(1).to(dtype=torch.float64)[:, None]
    gate_sink_mean = (gate_sink_sum / sink_den).float()
    gate_control_mean = (gate_control_sum / control_den).float()
    layer_mean_abs_gate = (abs_gate_all_sum / all_den).float().clamp_min(1e-8)
    contribution_sink_mean = (contribution_sink_sum / sink_den).float()
    contribution_control_mean = (contribution_control_sum / control_den).float().clamp_min(1e-8)

    direction_alignment: torch.Tensor | None = None
    if method == "gate_mean":
        scores = gate_sink_mean
    elif method == "layer_relative_gate":
        layer_scale = layer_mean_abs_gate.mean(dim=1, keepdim=True).clamp_min(1e-8)
        scores = gate_sink_mean / layer_scale
    elif method == "contribution_magnitude":
        layer_scale = contribution_sink_mean.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        scores = contribution_sink_mean / layer_scale
    elif method == "contribution_contrast":
        scores = contribution_sink_mean / contribution_control_mean
    elif method == "direction_aware":
        direction_alignment = torch.zeros((num_layers, intermediate_size), dtype=torch.float32)
        for layer_idx in range(top_layer + 1):
            down_proj = _score_projection_activation_and_up_down(layers[layer_idx].mlp)[3]
            if down_proj is None:
                continue
            sink_mean = sink_hidden_sum[layer_idx] / sink_counts[layer_idx].clamp_min(1).item()
            control_mean = control_hidden_sum[layer_idx] / control_counts[layer_idx].clamp_min(1).item()
            direction = (sink_mean - control_mean).float()
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
            columns = _down_columns(down_proj, intermediate_size)
            columns = columns / torch.linalg.vector_norm(columns, dim=1, keepdim=True).clamp_min(1e-8)
            direction_alignment[layer_idx] = columns @ direction
        scores = contribution_sink_mean * direction_alignment.clamp_min(0.0)
    else:
        raise ValueError(f"Unsupported score method: {method}")

    if top_layer + 1 < num_layers:
        scores[top_layer + 1 :] = float("-inf")

    return AblationScoreResult(
        scores=scores.float(),
        method=method,
        sink_counts_by_layer=sink_counts,
        control_counts_by_layer=control_counts,
        layer_mean_abs_gate=layer_mean_abs_gate,
        gate_sink_mean=gate_sink_mean,
        gate_control_mean=gate_control_mean,
        contribution_sink_mean=contribution_sink_mean,
        contribution_control_mean=contribution_control_mean,
        direction_alignment=direction_alignment,
    )
