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
MOVE = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_position_sweep" / "position_sweep_metrics.csv"
COPY = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_position_sweep_copy" / "position_sweep_metrics.csv"
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
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[float, dict[str, float]]:
    grouped: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pos = row.get("insert_after_pos")
        if pos is None or float(pos) < 0:
            continue
        condition = row["condition"]
        if condition == "selected_relocate":
            prefix = "selected"
        elif condition == "random_relocate":
            prefix = "random"
        elif condition == "dummy_only":
            prefix = "dummy_only"
        else:
            continue
        grouped[float(pos)][f"{prefix}_dummy_attn"].append(float(row["dummy_attention_mean"]))
        grouped[float(pos)][f"{prefix}_ppl_ratio"].append(float(row["ppl_ratio_vs_clean"]))
        grouped[float(pos)][f"{prefix}_bos_attn"].append(float(row["bos_attention_mean"]))
    out: dict[float, dict[str, float]] = {}
    for pos, vals in grouped.items():
        out[pos] = {key: float(np.mean(val)) for key, val in vals.items()}
    return out


def series(data: dict[float, dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array(sorted(data), dtype=float)
    ys = np.array([data[x][key] for x in xs], dtype=float)
    return xs, ys


def main() -> None:
    setup_style()
    move = aggregate(read_rows(MOVE))
    copy = aggregate(read_rows(COPY))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.78, wspace=0.36)

    x_copy, selected_copy = series(copy, "selected_dummy_attn")
    _, random_copy = series(copy, "random_dummy_attn")
    _, dummy_copy = series(copy, "dummy_only_dummy_attn")
    x_move, selected_move = series(move, "selected_dummy_attn")
    _, ppl_copy = series(copy, "selected_ppl_ratio")
    _, ppl_random = series(copy, "random_ppl_ratio")
    _, ppl_dummy = series(copy, "dummy_only_ppl_ratio")
    _, ppl_move = series(move, "selected_ppl_ratio")
    _, bos_copy = series(copy, "selected_bos_attn")
    _, bos_random = series(copy, "random_bos_attn")

    axes[0].plot(x_copy, selected_copy, color="#d1495b", marker="o", linewidth=2.5, label="Copy intervention", zorder=4)
    axes[0].plot(x_copy, random_copy, color="#3b528b", marker="o", linewidth=2.0, label="Random control", zorder=4)
    axes[0].plot(x_copy, dummy_copy, color="#6b7280", marker="o", linewidth=1.8, linestyle=":", label="Dummy only", zorder=3)

    axes[1].plot(x_copy, ppl_copy, color="#d1495b", marker="o", linewidth=2.5, label="Copy intervention", zorder=4)
    axes[1].plot(x_copy, ppl_random, color="#3b528b", marker="o", linewidth=2.0, label="Random control", zorder=4)
    axes[1].plot(x_copy, ppl_dummy, color="#6b7280", marker="o", linewidth=1.8, linestyle=":", label="Dummy only", zorder=3)
    axes[1].plot(x_move, ppl_move, color="#d1495b", marker="s", linewidth=1.9, linestyle="--", alpha=0.70, label="Move intervention", zorder=3)
    axes[1].axhline(1.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)

    axes[2].plot(x_copy, bos_copy, color="#d1495b", marker="o", linewidth=2.5, label="Copy intervention", zorder=4)
    axes[2].plot(x_copy, bos_random, color="#3b528b", marker="o", linewidth=2.0, label="Random control", zorder=4)

    specs = [
        (axes[0], "Inserted-Slot Attention", "Insertion Position", "Mean Attention"),
        (axes[1], "Perplexity Impact", "Insertion Position", "PPL Ratio"),
        (axes[2], "BOS Attention Displacement", "Insertion Position", "Mean BOS Attention"),
    ]
    for ax, title, xlabel, ylabel in specs:
        ax.set_title(title, fontsize=22, pad=10, fontweight=200)
        ax.set_xlabel(xlabel, fontsize=18, fontweight=200, labelpad=9)
        ax.set_ylabel(ylabel, fontsize=18, fontweight=200)
        ax.set_xscale("symlog", linthresh=8)
        ax.set_xticks([0, 8, 32, 128, 512])
        ax.set_xticklabels(["0", "8", "32", "128", "512"])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)
    axes[0].set_ylim(0.0, 0.36)
    axes[2].set_ylim(0.0, 0.36)
    for ax in axes:
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#9ca3af", facecolor="white", framealpha=0.96, fontsize=11)
    fig.suptitle("Sink Hijacking Position Sweep", fontsize=26, fontweight=200, y=0.965)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sink_hijacking_position_sweep.png", dpi=300)
    fig.savefig(OUT / "sink_hijacking_position_sweep.pdf")
    plt.close(fig)
    print((OUT / "sink_hijacking_position_sweep.png").relative_to(ROOT))
    print((OUT / "sink_hijacking_position_sweep.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
