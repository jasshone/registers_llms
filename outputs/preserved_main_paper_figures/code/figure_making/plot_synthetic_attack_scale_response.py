from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "sink_hijacking_attack" / "pythia_1b_synthetic_local_fact_scale_response_summary" / "synthetic_attack_scale_response_summary.csv"
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


def rows_for(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return sorted([r for r in rows if r["condition"] == condition], key=lambda r: float(r["scale"]))


def series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    return np.array([float(r[f"{key}_mean"]) for r in rows]), np.array([float(r[f"{key}_sd"]) for r in rows])


def main() -> None:
    setup_style()
    all_rows = read_rows(SUMMARY)
    selected = rows_for(all_rows, "selected_copy")
    random = rows_for(all_rows, "random_copy_mean")
    dummy = rows_for(all_rows, "dummy_only")
    scales = np.array([float(r["scale"]) for r in selected])
    colors = {"selected": "#d1495b", "random": "#3b528b", "dummy": "#6b7280"}

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.21, top=0.78, wspace=0.34)
    err_kw = {"elinewidth": 1.2, "capsize": 3, "capthick": 1.2}

    specs = [
        (axes[0], "dummy_attention_mean", "Sink Mass", "Dummy Attn."),
        (axes[1], "margin_delta_vs_clean", "Answer Margin", "Delta"),
        (axes[2], "clean_correct_to_wrong", "Answer Flips", "Clean-correct -> wrong"),
    ]
    for ax, key, title, ylabel in specs:
        for label, rows, color in [("Selected", selected, colors["selected"]), ("Random", random, colors["random"]), ("Dummy", dummy, colors["dummy"] )]:
            y, yerr = series(rows, key)
            ax.errorbar(scales, y, yerr=yerr, marker="o", linewidth=2.4, markersize=5.0, color=color, label=label, **err_kw, zorder=4)
        ax.set_title(title, fontsize=21, pad=10, fontweight=200)
        ax.set_xlabel("Scale", fontsize=17, fontweight=200, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=17, fontweight=200)
        ax.set_xticks(scales)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        if key == "margin_delta_vs_clean":
            ax.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
        clean_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=True, fancybox=False, edgecolor="#9ca3af", facecolor="white", framealpha=0.96, fontsize=13, handlelength=1.8, columnspacing=1.2)
    fig.suptitle("Synthetic Sink Attack Scale Response", fontsize=25, fontweight=200, y=0.965)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "synthetic_sink_attack_scale_response.png", dpi=300)
    fig.savefig(OUT / "synthetic_sink_attack_scale_response.pdf")
    plt.close(fig)
    print((OUT / "synthetic_sink_attack_scale_response.png").relative_to(ROOT))
    print((OUT / "synthetic_sink_attack_scale_response.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
