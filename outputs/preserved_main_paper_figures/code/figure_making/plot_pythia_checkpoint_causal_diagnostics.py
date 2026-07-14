from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "pythia_checkpoint_causal_diagnostics" / "test_64w_summary" / "checkpoint_causal_summary.csv"
OUT = ROOT / "outputs" / "pythia_checkpoint_causal_diagnostics" / "test_64w_summary"


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / "fonts"
    font_path = font_dir / "HelveticaNeueLight.otf"
    bold_path = font_dir / "HelveticaNeueBold.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.weight"] = 200
        plt.rcParams["axes.labelweight"] = 200
        plt.rcParams["axes.titleweight"] = 200
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="y", which="major", labelsize=15, width=1.4, length=5)
    ax.tick_params(axis="x", which="major", labelsize=14, width=1.4, length=5, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.16, zorder=0)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def main() -> None:
    setup_style()
    rows = read_rows(SUMMARY)
    x = arr(rows, "step")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.78, wspace=0.34)

    axes[0].plot(x, arr(rows, "clean_bos_attention_mean"), color="#3b528b", marker="o", linewidth=2.5, zorder=4)
    axes[1].plot(x, arr(rows, "zero_bos_bos_retention"), color="#d1495b", marker="o", linewidth=2.5, label="Selected zero", zorder=4)
    axes[1].plot(x, arr(rows, "random_zero_global_bos_retention"), color="#3b528b", marker="o", linewidth=2.0, label="Random zero", zorder=4)
    axes[1].axhline(1.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    axes[2].plot(x, arr(rows, "relocate_dummy_minus_bos_mean"), color="#21918c", marker="o", linewidth=2.5, zorder=4)
    axes[2].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)

    specs = [
        (axes[0], "BOS Sink Strength", "Mean BOS Attention"),
        (axes[1], "BOS Retention After Zeroing", "Retention Ratio"),
        (axes[2], "Relocation to Dummy", "Dummy - BOS Attention"),
    ]
    for ax, title, ylabel in specs:
        ax.set_title(title, fontsize=22, pad=10, fontweight=200)
        ax.set_xlabel("Training Step", fontsize=18, fontweight=200, labelpad=9)
        ax.set_ylabel(ylabel, fontsize=18, fontweight=200)
        ax.set_xscale("log")
        ax.set_xticks([2000, 4000, 8000, 16000, 143000])
        ax.set_xticklabels(["2k", "4k", "8k", "16k", "143k"])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)
    axes[1].legend(loc="best", frameon=True, fancybox=False, edgecolor="#9ca3af", facecolor="white", framealpha=0.96, fontsize=11)
    fig.suptitle("Pythia-1B Checkpoint Causal Diagnostics", fontsize=26, fontweight=200, y=0.965)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pythia_1b_checkpoint_causal_diagnostics.png", dpi=300)
    fig.savefig(OUT / "pythia_1b_checkpoint_causal_diagnostics.pdf")
    plt.close(fig)
    print((OUT / "pythia_1b_checkpoint_causal_diagnostics.png").relative_to(ROOT))
    print((OUT / "pythia_1b_checkpoint_causal_diagnostics.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
