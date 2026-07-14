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
SUMMARY = RUN_DIR / "synthetic_local_fact_3seed_summary.csv"
FLIPS = RUN_DIR / "synthetic_local_fact_3seed_flip_summary.csv"
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


def by_condition(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["condition"]): row for row in rows}


def main() -> None:
    setup_style()
    summary = by_condition(read_rows(SUMMARY))
    flips = by_condition(read_rows(FLIPS))
    conditions = ["dummy_only", "selected_copy", "random_copy_mean"]
    flip_keys = ["dummy_only", "selected_copy", "random_copy"]
    labels = ["Dummy", "Selected", "Random"]
    colors = ["#6b7280", "#d1495b", "#3b528b"]
    x = np.arange(len(labels))

    dummy_attn = np.array([float(summary[c]["dummy_attention_mean_mean"]) for c in conditions])
    dummy_attn_sd = np.array([float(summary[c]["dummy_attention_mean_sd"]) for c in conditions])
    acc_delta = np.array([100.0 * float(summary[c]["accuracy_delta_vs_clean_mean"]) for c in conditions])
    acc_delta_sd = np.array([100.0 * float(summary[c]["accuracy_delta_vs_clean_sd"]) for c in conditions])
    margin_delta = np.array([float(summary[c]["margin_delta_vs_clean_mean"]) for c in conditions])
    margin_delta_sd = np.array([float(summary[c]["margin_delta_vs_clean_sd"]) for c in conditions])
    clean_flips = np.array([float(flips[c]["clean_correct_to_wrong_mean"]) for c in flip_keys])
    clean_flips_sd = np.array([float(flips[c]["clean_correct_to_wrong_sd"]) for c in flip_keys])

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.78, wspace=0.36)

    err_kw = {"ecolor": "#111827", "elinewidth": 1.2, "capsize": 3, "capthick": 1.2}
    axes[0].bar(x, dummy_attn, yerr=dummy_attn_sd, error_kw=err_kw, color=colors, zorder=4)
    axes[1].bar(x, acc_delta, yerr=acc_delta_sd, error_kw=err_kw, color=colors, zorder=4)
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    axes[2].bar(x, margin_delta, yerr=margin_delta_sd, error_kw=err_kw, color=colors, zorder=4)
    axes[2].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    axes[3].bar(x, clean_flips, yerr=clean_flips_sd, error_kw=err_kw, color=colors, zorder=4)

    specs = [
        (axes[0], "Sink Mass", "Dummy Attn."),
        (axes[1], "Accuracy", "Delta (pp)"),
        (axes[2], "Answer Margin", "Delta"),
        (axes[3], "Answer Flips", "Clean-correct -> wrong"),
    ]
    for ax, title, ylabel in specs:
        ax.set_title(title, fontsize=21, pad=10, fontweight=200)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=17, fontweight=200)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)
    fig.suptitle("Synthetic Local-Fact Sink Attack", fontsize=25, fontweight=200, y=0.965)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "synthetic_local_fact_sink_attack.png", dpi=300)
    fig.savefig(OUT / "synthetic_local_fact_sink_attack.pdf")
    plt.close(fig)
    print((OUT / "synthetic_local_fact_sink_attack.png").relative_to(ROOT))
    print((OUT / "synthetic_local_fact_sink_attack.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
