from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "pythia_two_stage_emergence_summary" / "two_stage_timing_summary.csv"
OUT = ROOT / "outputs" / "pythia_two_stage_emergence_summary"


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
            if value == "":
                continue
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
    labels = [str(r["model"]) for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.80, wspace=0.30)

    width = 0.24
    axes[0].bar(x - width, arr(rows, "half_bos_step"), width=width, color="#3b528b", label="BOS attention", zorder=4)
    axes[0].bar(x, arr(rows, "half_selected_score_mass_step"), width=width, color="#21918c", label="Score mass", zorder=4)
    axes[0].bar(x + width, arr(rows, "half_selected_active_step"), width=width, color="#d1495b", label="Active neurons", zorder=4)

    lag_active = arr(rows, "lag_half_active_minus_half_bos") / 1000.0
    lag_mass = arr(rows, "lag_half_score_mass_minus_half_bos") / 1000.0
    axes[1].bar(x - width/2, lag_mass, width=width, color="#21918c", label="Score-mass lag", zorder=4)
    axes[1].bar(x + width/2, lag_active, width=width, color="#d1495b", label="Active-neuron lag", zorder=4)
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)

    axes[0].set_title("Timing", fontsize=23, pad=10, fontweight=200)
    axes[0].set_ylabel("Step", fontsize=18, fontweight=200)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1500, 160000)
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))

    axes[1].set_title("Lag vs BOS", fontsize=23, pad=10, fontweight=200)
    axes[1].set_ylabel("Lag (k)", fontsize=18, fontweight=200)
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(5))

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Pythia Model", fontsize=18, fontweight=200, labelpad=9)
        clean_axis(ax)
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#9ca3af", facecolor="white", framealpha=0.96, fontsize=11)

    fig.suptitle("Sink Emergence", fontsize=26, fontweight=200, y=0.965)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pythia_two_stage_emergence.png", dpi=300)
    fig.savefig(OUT / "pythia_two_stage_emergence.pdf")
    plt.close(fig)
    print((OUT / "pythia_two_stage_emergence.png").relative_to(ROOT))
    print((OUT / "pythia_two_stage_emergence.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
