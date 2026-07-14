from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATTACK = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_focused_best"
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
    ax.tick_params(axis="x", which="major", labelsize=14, width=1.4, length=0, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.16, zorder=0)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for k, v in list(row.items()):
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return float(np.mean(vals))


def main() -> None:
    setup_style()
    win = read_rows(ATTACK / "window_attack_metrics.csv")
    fact = read_rows(ATTACK / "local_fact_attack_metrics.csv")
    pos = "after_bos"
    win_clean = next(r for r in win if r["condition"] == "clean")
    win_dummy = next(r for r in win if r["condition"] == "dummy_only" and r["dummy_position"] == pos)
    win_sel = next(r for r in win if r["condition"] == "selected_relocate" and r["dummy_position"] == pos)
    win_rand = [r for r in win if r["condition"] == "random_relocate" and r["dummy_position"] == pos]
    fact_clean = next(r for r in fact if r["condition"] == "clean")
    fact_dummy = next(r for r in fact if r["condition"] == "dummy_only" and r["dummy_position"] == pos)
    fact_sel = next(r for r in fact if r["condition"] == "selected_relocate" and r["dummy_position"] == pos)
    fact_rand = [r for r in fact if r["condition"] == "random_relocate" and r["dummy_position"] == pos]

    labels = ["Clean", "Dummy", "Random", "Selected"]
    colors = ["#6b7280", "#9ca3af", "#21918c", "#d1495b"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.13, top=0.90, hspace=0.45, wspace=0.30)

    sink_vals = [0.0, float(win_dummy["dummy_attention_mean"]), mean(win_rand, "dummy_attention_mean"), float(win_sel["dummy_attention_mean"])]
    ppl_vals = [1.0, float(win_dummy["ppl_ratio_vs_clean"]), mean(win_rand, "ppl_ratio_vs_clean"), float(win_sel["ppl_ratio_vs_clean"])]
    margin_vals = [0.0, float(fact_dummy["margin_delta_vs_clean"]), mean(fact_rand, "margin_delta_vs_clean"), float(fact_sel["margin_delta_vs_clean"])]
    evidence_vals = [0.0, float(fact_dummy["evidence_attention_delta_vs_clean"]), mean(fact_rand, "evidence_attention_delta_vs_clean"), float(fact_sel["evidence_attention_delta_vs_clean"])]

    panels = [
        (axes[0, 0], sink_vals, "Dummy Attention Mass", "Mean Attention to Dummy Slots", None),
        (axes[0, 1], ppl_vals, "Perplexity Degradation", "PPL Ratio vs. Clean", 1.0),
        (axes[1, 0], margin_vals, "Local-Fact Logit Margin", "Margin Change vs. Clean", 0.0),
        (axes[1, 1], evidence_vals, "Evidence Attention", "Attention Change vs. Clean", 0.0),
    ]
    for ax, vals, title, ylabel, baseline in panels:
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.9, zorder=4)
        if baseline is not None:
            ax.axhline(baseline, color="#374151", linewidth=1.0, alpha=0.65, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=24, pad=10, fontweight=200)
        ax.set_ylabel(ylabel, fontsize=19, fontweight=200)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)

    fig.suptitle("Sink Hijacking via Selected MLP Neuron Relocation", fontsize=27, fontweight=200, y=0.985)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sink_hijacking_attack.png", dpi=300)
    fig.savefig(OUT / "sink_hijacking_attack.pdf")
    plt.close(fig)
    print((OUT / "sink_hijacking_attack.png").relative_to(ROOT))
    print((OUT / "sink_hijacking_attack.pdf").relative_to(ROOT))

if __name__ == "__main__":
    main()
