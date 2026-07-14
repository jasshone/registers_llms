from __future__ import annotations

import math
from typing import Any

import torch


def summarize_nll(loss_sums: torch.Tensor) -> dict[str, Any]:
    mean_nll = float(loss_sums.mean().item())
    return {
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "tokens_scored": int(loss_sums.numel()),
    }


def recovery_fraction(policy_delta_nll: float, sliding_delta_nll: float) -> float | None:
    if sliding_delta_nll <= 0:
        return None
    return 1.0 - (policy_delta_nll / sliding_delta_nll)

