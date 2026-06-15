from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
from typing import Callable, Literal

import torch
from transformers import PreTrainedModel

from .modeling import get_embed_tokens, get_transformer_layers


def make_layer_selection_map(selected: torch.Tensor) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for layer_idx, neuron_idx in selected.tolist():
        grouped[int(layer_idx)].append(int(neuron_idx))
    return {
        layer_idx: torch.tensor(sorted(neurons), dtype=torch.long)
        for layer_idx, neurons in grouped.items()
    }


DummyInit = Literal["zero", "bos"]
AssignmentStrategy = Literal["round_robin", "layer_blocked", "magnitude_balanced"]
DummyPosition = Literal["after_bos", "before_bos"]
InterventionBackend = Literal["mlp", "qwen3_moe"]


def make_layer_assignment_map(
    selected: torch.Tensor,
    *,
    num_dummy_tokens: int,
    assignment_strategy: AssignmentStrategy = "round_robin",
    selected_scores: torch.Tensor | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if num_dummy_tokens < 1:
        raise ValueError("num_dummy_tokens must be at least 1")
    if selected_scores is not None and selected_scores.numel() != selected.shape[0]:
        raise ValueError("selected_scores must contain one score per selected neuron")

    entries: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if assignment_strategy == "round_robin":
        grouped = make_layer_selection_map(selected)
        for layer_idx, neurons in grouped.items():
            assignments = torch.arange(neurons.numel(), dtype=torch.long) % num_dummy_tokens
            for neuron_idx, dummy_idx in zip(neurons.tolist(), assignments.tolist()):
                entries[layer_idx].append((int(neuron_idx), int(dummy_idx)))
    elif assignment_strategy == "layer_blocked":
        for layer_idx, neuron_idx in selected.tolist():
            entries[int(layer_idx)].append((int(neuron_idx), int(layer_idx) % num_dummy_tokens))
    elif assignment_strategy == "magnitude_balanced":
        scores = selected_scores.float().abs() if selected_scores is not None else torch.ones(selected.shape[0])
        loads = torch.zeros(num_dummy_tokens, dtype=torch.float64)
        for row_idx in torch.argsort(scores, descending=True).tolist():
            layer_idx = int(selected[row_idx, 0].item())
            neuron_idx = int(selected[row_idx, 1].item())
            dummy_idx = int(torch.argmin(loads).item())
            entries[layer_idx].append((neuron_idx, dummy_idx))
            loads[dummy_idx] += float(scores[row_idx].item())
    else:
        raise ValueError(f"Unsupported assignment_strategy: {assignment_strategy}")

    return {
        layer_idx: (
            torch.tensor([neuron for neuron, _dummy in sorted(layer_entries)], dtype=torch.long),
            torch.tensor([dummy for _neuron, dummy in sorted(layer_entries)], dtype=torch.long),
        )
        for layer_idx, layer_entries in entries.items()
    }




def _moe_intermediate_size(mlp: torch.nn.Module) -> int:
    if hasattr(mlp, "experts") and hasattr(mlp.experts, "intermediate_dim"):
        return int(mlp.experts.intermediate_dim)
    if hasattr(mlp, "config") and hasattr(mlp.config, "moe_intermediate_size"):
        return int(mlp.config.moe_intermediate_size)
    raise AttributeError("Unsupported MoE architecture: cannot infer expert intermediate size")


def _decode_moe_flat_neurons(flat_neurons: torch.Tensor, expert_intermediate: int) -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.div(flat_neurons, expert_intermediate, rounding_mode="floor")
    neurons = flat_neurons % expert_intermediate
    return experts.to(dtype=torch.long), neurons.to(dtype=torch.long)


def _make_qwen3_moe_forward(
    layer_mlp: torch.nn.Module,
    selected_flat_neurons: torch.Tensor,
    dummy_assignments: torch.Tensor,
    *,
    num_dummy_tokens: int,
    relocation_scale: float,
    relocation_fraction: float,
    dummy_position: DummyPosition,
) -> Callable:
    experts_module = layer_mlp.experts
    expert_intermediate = _moe_intermediate_size(layer_mlp)
    selected_experts, selected_neurons = _decode_moe_flat_neurons(selected_flat_neurons, expert_intermediate)
    by_expert: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for expert_idx in torch.unique(selected_experts).tolist():
        mask = selected_experts == int(expert_idx)
        by_expert[int(expert_idx)] = (
            selected_neurons[mask].to(dtype=torch.long),
            dummy_assignments[mask].to(dtype=torch.long),
            selected_flat_neurons[mask].to(dtype=torch.long),
        )

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
        _, routing_weights, selected_route_experts = layer_mlp.gate(hidden_states_reshaped)
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(selected_route_experts, num_classes=experts_module.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        if dummy_position == "after_bos":
            dummy_start = 1
        elif dummy_position == "before_bos":
            dummy_start = 0
        else:
            raise ValueError(f"Unsupported dummy_position: {dummy_position}")

        for expert_hit_idx in expert_hit:
            expert_idx = int(expert_hit_idx[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            gate_up = torch.nn.functional.linear(current_state, experts_module.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            gate_act = experts_module.act_fn(gate)

            if expert_idx in by_expert:
                local_neurons, local_assignments, _local_flat = by_expert[expert_idx]
                local_neurons = local_neurons.to(gate_act.device)
                local_assignments = local_assignments.to(gate_act.device)
                gate_selected = gate_act[:, local_neurons].clone()
                token_batch = torch.div(token_idx, sequence_length, rounding_mode="floor")
                token_pos = token_idx % sequence_length
                original_rows = torch.ones_like(token_pos, dtype=torch.bool)
                original_rows &= ~((token_pos >= dummy_start) & (token_pos < dummy_start + num_dummy_tokens))

                max_vals = torch.zeros((batch_size, local_neurons.numel()), dtype=gate_selected.dtype, device=gate_selected.device)
                for batch_idx in range(batch_size):
                    batch_rows = original_rows & (token_batch == batch_idx)
                    if batch_rows.any():
                        max_vals[batch_idx] = gate_selected[batch_rows].amax(dim=0)

                gate_selected[original_rows] = gate_selected[original_rows] * (1.0 - relocation_fraction)
                for dummy_idx in range(num_dummy_tokens):
                    dummy_rows = token_pos == (dummy_start + dummy_idx)
                    if not dummy_rows.any():
                        continue
                    assigned = local_assignments == dummy_idx
                    if not assigned.any():
                        continue
                    rows = torch.nonzero(dummy_rows, as_tuple=False).squeeze(1)
                    gate_selected[rows[:, None], torch.nonzero(assigned, as_tuple=False).squeeze(1)[None, :]] = (
                        max_vals[token_batch[rows]][:, assigned] * relocation_scale * relocation_fraction
                    )
                gate_act = gate_act.clone()
                gate_act[:, local_neurons] = gate_selected

            current_hidden_states = gate_act * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, experts_module.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return forward

def build_intervened_inputs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    dummy_init: DummyInit = "zero",
    num_dummy_tokens: int = 1,
    dummy_position: DummyPosition = "after_bos",
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_dummy_tokens < 1:
        raise ValueError("num_dummy_tokens must be at least 1")
    embed_tokens = get_embed_tokens(model)
    input_embeds = embed_tokens(input_ids)
    if dummy_init == "zero":
        batch, _, hidden = input_embeds.shape
        dummy_slot = torch.zeros(
            (batch, num_dummy_tokens, hidden),
            dtype=input_embeds.dtype,
            device=input_embeds.device,
        )
    elif dummy_init == "bos":
        dummy_slot = input_embeds[:, :1, :].expand(-1, num_dummy_tokens, -1).clone()
    else:
        raise ValueError(f"Unsupported dummy_init: {dummy_init}")
    if dummy_position == "after_bos":
        intervened_embeds = torch.cat([input_embeds[:, :1, :], dummy_slot, input_embeds[:, 1:, :]], dim=1)
    elif dummy_position == "before_bos":
        intervened_embeds = torch.cat([dummy_slot, input_embeds], dim=1)
    else:
        raise ValueError(f"Unsupported dummy_position: {dummy_position}")
    attention_mask = torch.ones(intervened_embeds.shape[:2], dtype=torch.long, device=input_ids.device)
    return intervened_embeds, attention_mask


def _split_mlp_forward(mlp: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    if all(hasattr(mlp, attr) for attr in ("gate_up_proj", "down_proj", "activation_fn")):
        gate, up = mlp.gate_up_proj(x).chunk(2, dim=-1)
        gate_act = mlp.activation_fn(gate)
        return gate_act, lambda updated_gate_act: mlp.down_proj(up * updated_gate_act)
    if all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        gate_act = mlp.act_fn(mlp.gate_proj(x))
        up = mlp.up_proj(x)
        return gate_act, lambda updated_gate_act: mlp.down_proj(updated_gate_act * up)
    if all(hasattr(mlp, attr) for attr in ("c_fc", "c_proj", "act")):
        hidden = mlp.act(mlp.c_fc(x))
        return hidden, mlp.c_proj
    if all(hasattr(mlp, attr) for attr in ("dense_h_to_4h", "dense_4h_to_h", "act")):
        hidden = mlp.act(mlp.dense_h_to_4h(x))
        return hidden, mlp.dense_4h_to_h
    if all(hasattr(mlp, attr) for attr in ("fc1", "fc2", "activation_fn")):
        hidden = mlp.activation_fn(mlp.fc1(x))
        return hidden, mlp.fc2
    raise AttributeError("Unsupported MLP architecture for relocation intervention")


def make_mlp_reroute_hook(
    model: PreTrainedModel,
    layer_idx: int,
    selected_neurons: torch.Tensor,
    dummy_assignments: torch.Tensor | None = None,
    num_dummy_tokens: int = 1,
    relocation_scale: float = 1.0,
    relocation_fraction: float = 1.0,
    dummy_position: DummyPosition = "after_bos",
) -> Callable:
    if num_dummy_tokens < 1:
        raise ValueError("num_dummy_tokens must be at least 1")
    if not 0.0 <= relocation_fraction <= 1.0:
        raise ValueError("relocation_fraction must be in [0, 1]")
    if dummy_assignments is None:
        dummy_assignments = torch.arange(selected_neurons.numel(), dtype=torch.long) % num_dummy_tokens
    layer = get_transformer_layers(model)[layer_idx]
    mlp = layer.mlp

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        if x.shape[1] < 1 + num_dummy_tokens:
            raise ValueError("Intervened sequence must contain at least BOS and DUMMY positions")

        selected_on_device = selected_neurons.to(x.device)
        gate_act, project = _split_mlp_forward(mlp, x)
        gate_selected = gate_act[..., selected_on_device].clone()

        original_positions = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        if dummy_position == "after_bos":
            dummy_start = 1
        elif dummy_position == "before_bos":
            dummy_start = 0
        else:
            raise ValueError(f"Unsupported dummy_position: {dummy_position}")
        original_positions[dummy_start : dummy_start + num_dummy_tokens] = False
        original_selected = gate_selected[:, original_positions, :]
        max_vals = original_selected.amax(dim=1)

        gate_selected[:, original_positions, :] = original_selected * (1.0 - relocation_fraction)
        assignments = dummy_assignments.to(x.device)
        for dummy_idx in range(num_dummy_tokens):
            assigned = assignments == dummy_idx
            gate_selected[:, dummy_start + dummy_idx, assigned] = max_vals[:, assigned] * relocation_scale * relocation_fraction

        gate_act = gate_act.clone()
        gate_act[..., selected_on_device] = gate_selected
        return project(gate_act)

    return hook


def register_mlp_reroute_hooks(
    model: PreTrainedModel,
    selected: torch.Tensor,
    *,
    num_dummy_tokens: int = 1,
    relocation_scale: float = 1.0,
    relocation_fraction: float = 1.0,
    assignment_strategy: AssignmentStrategy = "round_robin",
    selected_scores: torch.Tensor | None = None,
    dummy_position: DummyPosition = "after_bos",
    intervention_backend: InterventionBackend = "mlp",
) -> list[torch.utils.hooks.RemovableHandle]:
    if intervention_backend != "mlp":
        raise ValueError("register_mlp_reroute_hooks only supports intervention_backend='mlp'; use patched_mlp_reroute_forwards for MoE")
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, (neurons, assignments) in make_layer_assignment_map(
        selected,
        num_dummy_tokens=num_dummy_tokens,
        assignment_strategy=assignment_strategy,
        selected_scores=selected_scores,
    ).items():
        hook = make_mlp_reroute_hook(
            model,
            layer_idx,
            neurons,
            assignments,
            num_dummy_tokens=num_dummy_tokens,
            relocation_scale=relocation_scale,
            relocation_fraction=relocation_fraction,
            dummy_position=dummy_position,
        )
        handle = get_transformer_layers(model)[layer_idx].mlp.register_forward_hook(hook)
        handles.append(handle)
    return handles


@contextmanager
def patched_mlp_reroute_forwards(
    model: PreTrainedModel,
    selected: torch.Tensor,
    *,
    num_dummy_tokens: int = 1,
    relocation_scale: float = 1.0,
    relocation_fraction: float = 1.0,
    assignment_strategy: AssignmentStrategy = "round_robin",
    selected_scores: torch.Tensor | None = None,
    dummy_position: DummyPosition = "after_bos",
    intervention_backend: InterventionBackend = "mlp",
):
    """Patch selected MLP forwards so relocation does not run each selected MLP twice."""
    if num_dummy_tokens < 1:
        raise ValueError("num_dummy_tokens must be at least 1")
    if not 0.0 <= relocation_fraction <= 1.0:
        raise ValueError("relocation_fraction must be in [0, 1]")
    originals: dict[int, Callable] = {}

    try:
        assignment_map = make_layer_assignment_map(
            selected,
            num_dummy_tokens=num_dummy_tokens,
            assignment_strategy=assignment_strategy,
            selected_scores=selected_scores,
        )
        for layer_idx, (neurons, assignments) in assignment_map.items():
            layer = get_transformer_layers(model)[layer_idx]
            mlp = layer.mlp
            originals[layer_idx] = mlp.forward

            if intervention_backend == "qwen3_moe":
                mlp.forward = _make_qwen3_moe_forward(
                    mlp,
                    neurons,
                    assignments,
                    num_dummy_tokens=num_dummy_tokens,
                    relocation_scale=relocation_scale,
                    relocation_fraction=relocation_fraction,
                    dummy_position=dummy_position,
                )  # type: ignore[method-assign]
                continue
            if intervention_backend != "mlp":
                raise ValueError(f"Unsupported intervention_backend: {intervention_backend}")

            def make_forward(
                layer_mlp: torch.nn.Module,
                selected_neurons: torch.Tensor,
                dummy_assignments: torch.Tensor,
            ) -> Callable:
                def forward(x: torch.Tensor) -> torch.Tensor:
                    if x.shape[1] < 1 + num_dummy_tokens:
                        raise ValueError("Intervened sequence must contain at least BOS and DUMMY positions")

                    selected_on_device = selected_neurons.to(x.device)
                    gate_act, project = _split_mlp_forward(layer_mlp, x)
                    gate_selected = gate_act[..., selected_on_device].clone()

                    original_positions = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                    if dummy_position == "after_bos":
                        dummy_start = 1
                    elif dummy_position == "before_bos":
                        dummy_start = 0
                    else:
                        raise ValueError(f"Unsupported dummy_position: {dummy_position}")
                    original_positions[dummy_start : dummy_start + num_dummy_tokens] = False
                    original_selected = gate_selected[:, original_positions, :]
                    max_vals = original_selected.amax(dim=1)

                    gate_selected[:, original_positions, :] = original_selected * (1.0 - relocation_fraction)
                    assignments_on_device = dummy_assignments.to(x.device)
                    for dummy_idx in range(num_dummy_tokens):
                        assigned = assignments_on_device == dummy_idx
                        gate_selected[:, dummy_start + dummy_idx, assigned] = (
                            max_vals[:, assigned] * relocation_scale * relocation_fraction
                        )

                    gate_act = gate_act.clone()
                    gate_act[..., selected_on_device] = gate_selected
                    return project(gate_act)

                return forward

            mlp.forward = make_forward(mlp, neurons, assignments)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            get_transformer_layers(model)[layer_idx].mlp.forward = original  # type: ignore[method-assign]


@contextmanager
def patched_mlp_zero_forwards(model: PreTrainedModel, selected: torch.Tensor):
    """Patch selected MLP forwards so selected gate activations are zeroed in-place."""
    originals: dict[int, Callable] = {}

    try:
        for layer_idx, neurons in make_layer_selection_map(selected).items():
            layer = get_transformer_layers(model)[layer_idx]
            mlp = layer.mlp
            originals[layer_idx] = mlp.forward

            def make_forward(layer_mlp: torch.nn.Module, selected_neurons: torch.Tensor) -> Callable:
                def forward(x: torch.Tensor) -> torch.Tensor:
                    selected_on_device = selected_neurons.to(x.device)
                    gate_act, project = _split_mlp_forward(layer_mlp, x)
                    gate_act = gate_act.clone()
                    gate_act[..., selected_on_device] = 0.0
                    return project(gate_act)

                return forward

            mlp.forward = make_forward(mlp, neurons)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            get_transformer_layers(model)[layer_idx].mlp.forward = original  # type: ignore[method-assign]
