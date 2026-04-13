from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch
from transformers import PreTrainedModel


def make_layer_selection_map(selected: torch.Tensor) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for layer_idx, neuron_idx in selected.tolist():
        grouped[int(layer_idx)].append(int(neuron_idx))
    return {
        layer_idx: torch.tensor(sorted(neurons), dtype=torch.long)
        for layer_idx, neurons in grouped.items()
    }


def build_intervened_inputs(model: PreTrainedModel, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    embed_tokens = model.model.embed_tokens
    input_embeds = embed_tokens(input_ids)
    batch, _, hidden = input_embeds.shape
    zero_slot = torch.zeros((batch, 1, hidden), dtype=input_embeds.dtype, device=input_embeds.device)
    intervened_embeds = torch.cat([input_embeds[:, :1, :], zero_slot, input_embeds[:, 1:, :]], dim=1)
    attention_mask = torch.ones(intervened_embeds.shape[:2], dtype=torch.long, device=input_ids.device)
    return intervened_embeds, attention_mask


def make_mlp_reroute_hook(
    model: PreTrainedModel,
    layer_idx: int,
    selected_neurons: torch.Tensor,
) -> Callable:
    layer = model.model.layers[layer_idx]

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        if x.shape[1] < 2:
            raise ValueError("Intervened sequence must contain at least BOS and DUMMY positions")

        selected_on_device = selected_neurons.to(x.device)
        gate_act = layer.mlp.act_fn(layer.mlp.gate_proj(x))
        up = layer.mlp.up_proj(x)
        gate_selected = gate_act[..., selected_on_device].clone()

        original_positions = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        original_positions[1] = False
        original_selected = gate_selected[:, original_positions, :]
        max_vals = original_selected.amax(dim=1)

        gate_selected[:, original_positions, :] = 0.0
        gate_selected[:, 1, :] = max_vals

        gate_act = gate_act.clone()
        gate_act[..., selected_on_device] = gate_selected
        return layer.mlp.down_proj(gate_act * up)

    return hook


def register_mlp_reroute_hooks(
    model: PreTrainedModel,
    selected: torch.Tensor,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, neurons in make_layer_selection_map(selected).items():
        hook = make_mlp_reroute_hook(model, layer_idx, neurons)
        handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
        handles.append(handle)
    return handles
