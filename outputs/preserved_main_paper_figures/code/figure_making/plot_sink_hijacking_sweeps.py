from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_representative"
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
    ax.tick_params(axis="x", which="major", labelsize=15, width=1.4, length=5, pad=8)
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


def select(rows, *, condition: str, pos: str, cap: int | None = None, scale: float | None = None):
    out = [r for r in rows if r["condition"] == condition and r.get("dummy_position") == pos]
    if cap is not None:
        out = [r for r in out if int(r["num_dummy_tokens"]) == cap]
    if scale is not None:
        out = [r for r in out if float(r["scale"]) == scale]
    return out


def main() -> None:
    setup_style()
    win = read_rows(REP / "window_attack_metrics.csv")
    fact = read_rows(REP / "local_fact_attack_metrics.csv")
    colors = {1: "#3b528b", 2: "#21918c", 4: "#5ec962"}
    scales = [1.0, 2.0, 4.0]
    caps = [1, 4, 8]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.14, top=0.90, hspace=0.45, wspace=0.30)

    # Scale sweep at capacity 8, after BOS.
    for cap in [1, 4, 8]:
        rows = [select(win, condition="selected_relocate", pos="after_bos", cap=cap, scale=s)[0] for s in scales]
        axes[0, 0].plot(scales, [float(r["dummy_attention_mean"]) for r in rows], color=colors.get(cap, "#111827"), marker="o", linewidth=2.4, label=f"{cap} slots", zorder=4)
        axes[0, 1].plot(scales, [float(r["ppl_ratio_vs_clean"]) for r in rows], color=colors.get(cap, "#111827"), marker="o", linewidth=2.4, label=f"{cap} slots", zorder=4)

    # Capacity sweep at scale 4, position comparison.
    for pos, color, label in [("after_bos", "#d1495b", "After BOS"), ("before_bos", "#21918c", "Before BOS")]:
        rows = [select(fact, condition="selected_relocate", pos=pos, cap=c, scale=4.0)[0] for c in caps]
        axes[1, 0].plot(caps, [float(r["margin_delta_vs_clean"]) for r in rows], color=color, marker="o", linewidth=2.4, label=label, zorder=4)
        axes[1, 1].plot(caps, [float(r["evidence_attention_delta_vs_clean"]) for r in rows], color=color, marker="o", linewidth=2.4, label=label, zorder=4)

    specs = [
        (axes[0, 0], "Scale Sweep: Dummy Sink Mass", "Relocation Scale", "Mean Attention to Dummy Slots"),
        (axes[0, 1], "Scale Sweep: Perplexity", "Relocation Scale", "PPL Ratio vs. Clean"),
        (axes[1, 0], "Capacity Sweep: Logit Margin", "Number of Dummy Slots", "Margin Change vs. Clean"),
        (axes[1, 1], "Capacity Sweep: Evidence Attention", "Number of Dummy Slots", "Attention Change vs. Clean"),
    ]
    for ax, title, xlabel, ylabel in specs:
        ax.set_title(title, fontsize=23, pad=10, fontweight=200)
        ax.set_xlabel(xlabel, fontsize=19, fontweight=200, labelpad=9)
        ax.set_ylabel(ylabel, fontsize=18, fontweight=200)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)
    axes[0, 1].axhline(1.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    axes[1, 0].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    axes[1, 1].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    for ax in axes.flat:
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#9ca3af", facecolor="white", framealpha=0.96, fontsize=12)
    fig.suptitle("Sink Hijacking Sweeps", fontsize=27, fontweight=200, y=0.985)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sink_hijacking_sweeps.png", dpi=300)
    fig.savefig(OUT / "sink_hijacking_sweeps.pdf")
    plt.close(fig)
    print((OUT / "sink_hijacking_sweeps.png").relative_to(ROOT))
    print((OUT / "sink_hijacking_sweeps.pdf").relative_to(ROOT))

if __name__ == "__main__":
    main()
