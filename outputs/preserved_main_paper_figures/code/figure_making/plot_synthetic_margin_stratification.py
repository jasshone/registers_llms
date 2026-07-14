from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_synthetic_local_fact_3seed_summary"
SUMMARY = RUN_DIR / "clean_margin_stratification_metrics.csv"
OUT = ROOT / "outputs" / "sink_hijacking_attack" / "figures"


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
    ax.tick_params(axis="y", which="major", labelsize=13, width=1.3, length=5)
    ax.tick_params(axis="x", which="major", labelsize=12, width=1.3, length=5, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.4)
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


def main() -> None:
    setup_style()
    rows = read_rows(SUMMARY)
    bins = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    bin_labels = ["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"]
    conditions = ["dummy_only", "selected_copy", "random_copy"]
    labels = ["Dummy", "Selected", "Random"]
    colors = ["#6b7280", "#d1495b", "#3b528b"]
    by = {(str(row["margin_bin"]), str(row["condition"])): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.80, wspace=0.28)
    width = 0.24
    x = np.arange(len(bins))

    for offset, cond, label, color in zip([-width, 0.0, width], conditions, labels, colors):
        flip_vals = [float(by[(b, cond)]["flip_rate_pct"]) for b in bins]
        margin_vals = [float(by[(b, cond)]["mean_margin_delta"]) for b in bins]
        axes[0].bar(x + offset, flip_vals, width=width, label=label, color=color, zorder=4)
        axes[1].bar(x + offset, margin_vals, width=width, label=label, color=color, zorder=4)

    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    specs = [
        (axes[0], "Flip Rate", "C->W rate (%)", (0, 52)),
        (axes[1], "Margin Damage", "Margin delta", (-1.55, 0.25)),
    ]
    for ax, title, ylabel, ylim in specs:
        ax.set_title(title, fontsize=21, pad=10, fontweight=200)
        ax.set_ylabel(ylabel, fontsize=16, fontweight=200)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        clean_axis(ax)
    axes[0].legend(frameon=False, fontsize=13, loc="upper right")
    fig.suptitle("Clean-Margin Stratification", fontsize=25, fontweight=200, y=0.96)
    fig.text(
        0.075,
        0.055,
        "Clean-correct examples binned by clean answer margin. Random combines three layer-matched controls per seed.",
        fontsize=12.5,
        color="#374151",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "synthetic_clean_margin_stratification.png", dpi=300)
    fig.savefig(OUT / "synthetic_clean_margin_stratification.pdf")
    plt.close(fig)
    print((OUT / "synthetic_clean_margin_stratification.png").relative_to(ROOT))
    print((OUT / "synthetic_clean_margin_stratification.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
