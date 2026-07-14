from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache

from sink_neurons.intervention import build_intervened_inputs

from .interventions import build_dummy_prefix_inputs, relocation_context
from .interventions import patched_mlp_relocate_to_position_forwards
from .metrics import summarize_nll
from .policies import context_for_policy


@torch.no_grad()
def evaluate_streaming_policy(
    model: PreTrainedModel,
    streams: torch.Tensor,
    *,
    policy: str,
    window_size: int,
    first_k: int = 4,
    selected: torch.Tensor | None = None,
    dummy_init: str = "bos",
    start_index: int = 1,
    max_positions: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    device = model.device
    stream_count, stream_length = streams.shape
    stop_index = stream_length if max_positions is None else min(stream_length, start_index + max_positions)
    losses: list[float] = []
    stream_indices: list[int] = []
    position_indices: list[int] = []

    total_steps = stream_count * max(0, stop_index - start_index)
    progress = tqdm(total=total_steps, desc=f"Streaming {policy}", disable=not show_progress, dynamic_ncols=True)
    for stream_idx, stream in enumerate(streams):
        stream = stream.to(device)
        for target_idx in range(start_index, stop_index):
            context = context_for_policy(
                stream,
                target_idx=target_idx,
                policy=policy,
                window_size=window_size,
                first_k=first_k,
            )
            labels = stream[target_idx].view(1)
            with relocation_context(model, selected, context.relocate_to_dummy):
                if context.uses_dummy:
                    inputs_embeds = build_dummy_prefix_inputs(
                        model,
                        context.token_ids.to(device),
                        dummy_init=dummy_init,  # type: ignore[arg-type]
                    )
                    outputs = model(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
                else:
                    input_ids = context.token_ids.to(device).unsqueeze(0)
                    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = outputs.logits[:, -1, :].float()
            loss = F.cross_entropy(logits, labels, reduction="none")
            losses.append(float(loss.item()))
            stream_indices.append(stream_idx)
            position_indices.append(target_idx)
            progress.update(1)
    progress.close()

    loss_tensor = torch.tensor(losses, dtype=torch.float32)
    return {
        "nll": loss_tensor,
        "stream_index": torch.tensor(stream_indices, dtype=torch.long),
        "position_index": torch.tensor(position_indices, dtype=torch.long),
        "summary": summarize_nll(loss_tensor),
        "num_streams": int(stream_count),
        "stream_length": int(stream_length),
        "start_index": int(start_index),
        "stop_index": int(stop_index),
    }


def _prune_dynamic_cache(past_key_values: DynamicCache, keep_indices: list[int]) -> None:
    if not keep_indices:
        for layer in past_key_values.layers:
            if layer.get_seq_length() > 0:
                layer.keys = layer.keys[..., :0, :]
                layer.values = layer.values[..., :0, :]
        return

    index = torch.tensor(keep_indices, dtype=torch.long)
    for layer in past_key_values.layers:
        if layer.get_seq_length() == 0:
            continue
        index_on_device = index.to(layer.keys.device)
        layer.keys = layer.keys.index_select(-2, index_on_device)
        layer.values = layer.values.index_select(-2, index_on_device)


def _retained_labels(
    labels: list[int | str],
    *,
    policy: str,
    window_size: int,
    first_k: int,
) -> set[int | str]:
    original_positions = [label for label in labels if isinstance(label, int)]
    max_original = max(original_positions) if original_positions else -1
    recent_start = max(1, max_original - window_size + 1)
    recent = {label for label in original_positions if label >= recent_start}

    if policy == "full_cache":
        return set(labels)
    if policy == "sliding_window":
        return {label for label in original_positions if label >= max(0, max_original - window_size + 1)}
    if policy == "streamingllm_first_k":
        return {label for label in original_positions if label < first_k} | recent
    if policy == "keep_bos_only":
        return {0} | recent
    if policy in {"dummy_no_relocation", "dummy_relocated"}:
        return {"dummy"} | recent
    raise ValueError(f"Unsupported streaming policy: {policy}")


def _prune_cache_for_policy(
    past_key_values: DynamicCache,
    labels: list[int | str],
    *,
    policy: str,
    window_size: int,
    first_k: int,
) -> list[int | str]:
    keep = _retained_labels(labels, policy=policy, window_size=window_size, first_k=first_k)
    keep_indices = [idx for idx, label in enumerate(labels) if label in keep]
    _prune_dynamic_cache(past_key_values, keep_indices)
    return [labels[idx] for idx in keep_indices]


def _cache_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.ones((1, length), dtype=torch.long, device=device)


def _position_id(position: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[position]], dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_streaming_policy_kv_cache(
    model: PreTrainedModel,
    streams: torch.Tensor,
    *,
    policy: str,
    window_size: int,
    first_k: int = 4,
    selected: torch.Tensor | None = None,
    dummy_init: str = "bos",
    start_index: int = 1,
    max_positions: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    device = model.device
    stream_count, stream_length = streams.shape
    stop_index = stream_length if max_positions is None else min(stream_length, start_index + max_positions)
    if stop_index <= start_index:
        raise ValueError("No target positions selected for scoring")

    losses: list[float] = []
    stream_indices: list[int] = []
    position_indices: list[int] = []
    total_steps = stream_count * max(0, stop_index - 1)
    progress = tqdm(total=total_steps, desc=f"KV streaming {policy}", disable=not show_progress, dynamic_ncols=True)

    for stream_idx, stream in enumerate(streams):
        stream = stream.to(device)
        past_key_values = DynamicCache(config=model.config)
        cache_labels: list[int | str] = []

        if policy in {"dummy_no_relocation", "dummy_relocated"}:
            inputs_embeds, attention_mask = build_intervened_inputs(
                model,
                stream[:1].view(1, 1),
                dummy_init=dummy_init,  # type: ignore[arg-type]
            )
            position_ids = torch.tensor([[0, 1]], dtype=torch.long, device=device)
            context = (
                patched_mlp_relocate_to_position_forwards(model, selected, dummy_position=1)
                if policy == "dummy_relocated" and selected is not None and int(selected.shape[0]) > 0
                else nullcontext()
            )
            with context:
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = outputs.past_key_values
            cache_labels = [0, "dummy"]
            cache_labels = _prune_cache_for_policy(
                past_key_values,
                cache_labels,
                policy=policy,
                window_size=window_size,
                first_k=first_k,
            )
            input_start = 1
        else:
            input_start = 0

        for input_pos in range(input_start, stop_index):
            cache_labels = _prune_cache_for_policy(
                past_key_values,
                cache_labels,
                policy=policy,
                window_size=window_size,
                first_k=first_k,
            )
            position = input_pos + 1 if policy in {"dummy_no_relocation", "dummy_relocated"} else input_pos
            input_ids = stream[input_pos].view(1, 1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=_cache_attention_mask(len(cache_labels) + 1, device),
                position_ids=_position_id(position, device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            cache_labels.append(input_pos)

            target_idx = input_pos + 1
            if start_index <= target_idx < stop_index:
                label = stream[target_idx].view(1)
                loss = F.cross_entropy(outputs.logits[:, -1, :].float(), label, reduction="none")
                losses.append(float(loss.item()))
                stream_indices.append(stream_idx)
                position_indices.append(target_idx)
            progress.update(1)

        cache_labels = _prune_cache_for_policy(
            past_key_values,
            cache_labels,
            policy=policy,
            window_size=window_size,
            first_k=first_k,
        )
    progress.close()

    loss_tensor = torch.tensor(losses, dtype=torch.float32)
    return {
        "nll": loss_tensor,
        "stream_index": torch.tensor(stream_indices, dtype=torch.long),
        "position_index": torch.tensor(position_indices, dtype=torch.long),
        "summary": summarize_nll(loss_tensor),
        "num_streams": int(stream_count),
        "stream_length": int(stream_length),
        "start_index": int(start_index),
        "stop_index": int(stop_index),
    }
