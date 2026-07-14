from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamContext:
    token_ids: torch.Tensor
    uses_dummy: bool = False
    relocate_to_dummy: bool = False


def _recent_start(target_idx: int, window_size: int) -> int:
    return max(1, target_idx - window_size)


def context_for_policy(
    stream: torch.Tensor,
    *,
    target_idx: int,
    policy: str,
    window_size: int,
    first_k: int = 4,
) -> StreamContext:
    if target_idx <= 0:
        raise ValueError("target_idx must be positive so there is context")

    if policy == "full_cache":
        return StreamContext(token_ids=stream[:target_idx])

    if policy == "sliding_window":
        start = max(0, target_idx - window_size)
        return StreamContext(token_ids=stream[start:target_idx])

    if policy == "streamingllm_first_k":
        prefix_end = min(first_k, target_idx)
        recent_start = max(prefix_end, target_idx - window_size)
        return StreamContext(token_ids=torch.cat([stream[:prefix_end], stream[recent_start:target_idx]]))

    if policy == "keep_bos_only":
        recent_start = _recent_start(target_idx, window_size)
        return StreamContext(token_ids=torch.cat([stream[:1], stream[recent_start:target_idx]]))

    if policy == "dummy_no_relocation":
        recent_start = _recent_start(target_idx, window_size)
        return StreamContext(token_ids=torch.cat([stream[:1], stream[recent_start:target_idx]]), uses_dummy=True)

    if policy == "dummy_relocated":
        recent_start = _recent_start(target_idx, window_size)
        return StreamContext(
            token_ids=torch.cat([stream[:1], stream[recent_start:target_idx]]),
            uses_dummy=True,
            relocate_to_dummy=True,
        )

    raise ValueError(f"Unsupported streaming policy: {policy}")


POLICIES = (
    "full_cache",
    "sliding_window",
    "streamingllm_first_k",
    "keep_bos_only",
    "dummy_no_relocation",
    "dummy_relocated",
)

