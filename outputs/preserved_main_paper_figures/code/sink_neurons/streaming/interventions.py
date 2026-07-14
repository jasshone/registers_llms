from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Callable

import torch
from transformers import PreTrainedModel

from sink_neurons.intervention import DummyInit, make_layer_selection_map


def build_dummy_prefix_inputs(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    *,
    dummy_init: DummyInit,
) -> torch.Tensor:
    embeds = model.model.embed_tokens(token_ids.unsqueeze(0))
    if dummy_init == "zero":
        dummy = torch.zeros_like(embeds[:, :1, :])
    elif dummy_init == "bos":
        dummy = embeds[:, :1, :].clone()
    else:
        raise ValueError(f"Unsupported dummy_init: {dummy_init}")
    return torch.cat([dummy, embeds[:, 1:, :]], dim=1)


@contextmanager
def patched_mlp_relocate_to_position_forwards(
    model: PreTrainedModel,
    selected: torch.Tensor,
    *,
    dummy_position: int,
):
    """Move selected MLP gate activations from non-dummy positions to dummy_position."""
    originals: dict[int, Callable] = {}

    try:
        for layer_idx, neurons in make_layer_selection_map(selected).items():
            layer = model.model.layers[layer_idx]
            mlp = layer.mlp
            originals[layer_idx] = mlp.forward

            def make_forward(layer_mlp: torch.nn.Module, selected_neurons: torch.Tensor) -> Callable:
                def forward(x: torch.Tensor) -> torch.Tensor:
                    if x.shape[1] <= dummy_position:
                        raise ValueError("Sequence is too short for requested dummy_position")
                    selected_on_device = selected_neurons.to(x.device)
                    gate_act = layer_mlp.act_fn(layer_mlp.gate_proj(x))
                    up = layer_mlp.up_proj(x)
                    gate_selected = gate_act[..., selected_on_device].clone()

                    original_positions = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                    original_positions[dummy_position] = False
                    if original_positions.any():
                        max_vals = gate_selected[:, original_positions, :].amax(dim=1)
                    else:
                        max_vals = gate_selected[:, dummy_position, :]
                    gate_selected[:, original_positions, :] = 0.0
                    gate_selected[:, dummy_position, :] = max_vals

                    gate_act = gate_act.clone()
                    gate_act[..., selected_on_device] = gate_selected
                    return layer_mlp.down_proj(gate_act * up)

                return forward

            mlp.forward = make_forward(mlp, neurons)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            model.model.layers[layer_idx].mlp.forward = original  # type: ignore[method-assign]


def relocation_context(model: PreTrainedModel, selected: torch.Tensor | None, enabled: bool):
    if enabled and selected is not None and int(selected.shape[0]) > 0:
        return patched_mlp_relocate_to_position_forwards(model, selected, dummy_position=0)
    return nullcontext()

