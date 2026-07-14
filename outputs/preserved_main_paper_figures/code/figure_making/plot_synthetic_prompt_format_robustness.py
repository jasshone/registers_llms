from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_synthetic_local_fact_3seed_summary"
SUMMARY = RUN_DIR / "prompt_format_robustness_metrics.csv"
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


def grouped(rows: list[dict[str, Any]], group_type: str) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["group_type"] != group_type:
            continue
        out[str(row["group"])][str(row["condition"])] = row
    return out


def main() -> None:
    setup_style()
    rows = read_rows(SUMMARY)
    evidence = grouped(rows, "prompt_family")
    prefix = grouped(rows, "prefix_family")

    evidence_order = [
        "label sentence",
        "target-letter sentence",
        "rule sentence",
        "correct-choice sentence",
        "evidence sentence",
        "answer-key fact",
    ]
    prefix_order = ["bare prompt", "instruction prefix", "metadata prefix"]
    conditions = ["dummy_only", "selected_copy", "random_copy"]
    labels = ["Dummy", "Selected", "Random"]
    colors = ["#6b7280", "#d1495b", "#3b528b"]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.78, wspace=0.26)

    width = 0.24
    for ax, data, order, title in [
        (axes[0], evidence, evidence_order, "Evidence Wording"),
        (axes[1], prefix, prefix_order, "Prompt Prefix"),
    ]:
        x = np.arange(len(order))
        for offset, cond, label, color in zip([-width, 0.0, width], conditions, labels, colors):
            vals = [float(data[group][cond]["flip_rate_clean_correct_pct"]) for group in order]
            ax.bar(x + offset, vals, width=width, label=label, color=color, zorder=4)
        ax.set_title(title, fontsize=21, pad=10, fontweight=200)
        ax.set_ylabel("C->W rate (%)", fontsize=16, fontweight=200)
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace(" sentence", "").replace("answer-key", "answer key") for g in order], rotation=28, ha="right")
        ax.set_ylim(0, 50)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        clean_axis(ax)

    axes[1].legend(frameon=False, fontsize=13, loc="upper right")
    fig.suptitle("Prompt-Format Robustness", fontsize=25, fontweight=200, y=0.96)
    fig.text(
        0.075,
        0.055,
        "Three Pythia-1B seeds; rates computed on clean-correct examples. Random combines three layer-matched controls per seed.",
        fontsize=12.5,
        color="#374151",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "synthetic_prompt_format_robustness.png", dpi=300)
    fig.savefig(OUT / "synthetic_prompt_format_robustness.pdf")
    plt.close(fig)
    print((OUT / "synthetic_prompt_format_robustness.png").relative_to(ROOT))
    print((OUT / "synthetic_prompt_format_robustness.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
