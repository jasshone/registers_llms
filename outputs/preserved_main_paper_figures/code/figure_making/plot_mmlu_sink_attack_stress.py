from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "sink_hijacking_attack" / "mmlu_copy_attack_100_summary" / "mmlu_copy_attack_100_summary.csv"
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
    ax.tick_params(axis="y", which="major", labelsize=15, width=1.4, length=5)
    ax.tick_params(axis="x", which="major", labelsize=13, width=1.4, length=5, pad=8)
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
    rows = [r for r in read_rows(SUMMARY) if r["condition"] != "clean"]
    labels = ["Dummy only", "Selected", "Random"]
    x = np.arange(len(rows))
    colors = ["#6b7280", "#d1495b", "#3b528b"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.78, wspace=0.34)

    axes[0].bar(x, arr(rows, "dummy_attention_mean"), color=colors, zorder=4)
    axes[1].bar(x, arr(rows, "accuracy_delta_vs_clean"), color=colors, zorder=4)
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    axes[2].bar(x, arr(rows, "choice_nll_delta_vs_clean"), color=colors, zorder=4)
    axes[2].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)

    specs = [
        (axes[0], "Inserted-Slot Attention", "Mean Attention"),
        (axes[1], "MMLU Accuracy", "Change vs. Clean"),
        (axes[2], "Choice NLL", "Change vs. Clean"),
    ]
    for ax, title, ylabel in specs:
        ax.set_title(title, fontsize=22, pad=10, fontweight=200)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=18, fontweight=200)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)
    fig.suptitle("MMLU Sink-Attack Stress Test", fontsize=26, fontweight=200, y=0.965)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mmlu_sink_attack_stress.png", dpi=300)
    fig.savefig(OUT / "mmlu_sink_attack_stress.pdf")
    plt.close(fig)
    print((OUT / "mmlu_sink_attack_stress.png").relative_to(ROOT))
    print((OUT / "mmlu_sink_attack_stress.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
