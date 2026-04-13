from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .env import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import torch


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mean_token_norm_heatmap(mean_token_norms: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_token_norms.cpu().numpy(), aspect="auto", origin="lower")
    ax.set_title("Mean Token Norm Heatmap")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax)
    _save(fig, path)


def plot_mean_bos_attention_by_layer(values: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values.cpu().numpy(), marker="o")
    ax.set_title("Mean BOS Attention by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Received")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_bos_score_histogram(scores: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores.reshape(-1).cpu().numpy(), bins=100)
    ax.set_title("BOS Neuron Score Histogram")
    ax.set_xlabel("Mean BOS Activation")
    ax.set_ylabel("Count")
    _save(fig, path)


def plot_neuron_count_vs_percentile(percentiles: Sequence[float], counts: Sequence[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(percentiles, counts, marker="o")
    ax.set_title("Neuron Count vs Percentile")
    ax.set_xlabel("Percentile Threshold")
    ax.set_ylabel("Selected Neuron Count")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_norm_relocation(
    bos_before: torch.Tensor,
    bos_after: torch.Tensor,
    dummy_after: torch.Tensor,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bos_before.cpu().numpy(), label="BOS before", marker="o")
    ax.plot(bos_after.cpu().numpy(), label="BOS after", marker="o")
    ax.plot(dummy_after.cpu().numpy(), label="Dummy after", marker="o")
    ax.set_title("Norm Relocation")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_attention_relocation(
    bos_before: torch.Tensor,
    bos_after: torch.Tensor,
    dummy_after: torch.Tensor,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bos_before.cpu().numpy(), label="BOS before", marker="o")
    ax.plot(bos_after.cpu().numpy(), label="BOS after", marker="o")
    ax.plot(dummy_after.cpu().numpy(), label="Dummy after", marker="o")
    ax.set_title("Attention Relocation")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Received")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_relocation_summary(
    x_values: Sequence[float],
    summary_values: Sequence[float],
    x_label: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_values, summary_values, marker="o")
    ax.set_title("Relocation Quality Summary")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean(dummy_attention_after - bos_attention_after)")
    ax.grid(True, alpha=0.3)
    _save(fig, path)
