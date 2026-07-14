from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "outputs" / "sink_hijacking_attack" / "squad_full_val_representatives"
MODEL_LABELS = {
    "gpt2_medium": "GPT-2 Medium",
    "pythia_2_8b": "Pythia 2.8B",
    "llama3_2_1b": "Llama 3.2 1B",
    "qwen3_1_7b": "Qwen3 1.7B",
    "mistral_7b_v0_1": "Mistral 7B",
}
ORDER = list(MODEL_LABELS)


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


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, val in list(row.items()):
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                pass
    return rows


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.yaxis.grid(True, color="#6b7280", linewidth=1.0, alpha=0.16, zorder=0)
    ax.tick_params(axis="y", labelsize=12, width=1.2, length=4)
    ax.tick_params(axis="x", labelsize=11, width=1.2, length=0, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)


def condition_map(rows: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if row.get("condition") == condition:
            out[str(row["model_key"])] = row
    return out


def flip_map(rows: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if row.get("condition") == condition:
            out[str(row["model_key"])] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--title", default="SQuAD Sink-Hijacking Attack")
    parser.add_argument("--stem", default="squad_sink_hijacking_representatives")
    args = parser.parse_args()
    src = args.src if args.src.is_absolute() else ROOT / args.src
    out = src / "figures"
    setup_style()
    summary_path = src / "combined_summary.csv"
    flips_path = src / "combined_flips.csv"
    if not summary_path.exists() or not flips_path.exists():
        raise SystemExit(f"missing combined outputs under {src}")
    rows = read_rows(summary_path)
    flips = read_rows(flips_path)
    present = [m for m in ORDER if any(str(r.get("model_key")) == m for r in rows)]
    labels = [MODEL_LABELS[m] for m in present]
    x = np.arange(len(present))
    width = 0.28

    clean = condition_map(rows, "clean")
    dummy = condition_map(rows, "dummy_only")
    selected = condition_map(rows, "selected_copy")
    random = condition_map(rows, "random_copy")
    flip_selected = flip_map(flips, "selected_copy")
    flip_random = flip_map(flips, "random_copy")

    selected_color = "#d1495b"
    random_color = "#21918c"
    dummy_color = "#4b5563"

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.80, wspace=0.32)

    acc_sel = [100 * float(selected[m]["accuracy_delta_vs_clean"]) for m in present]
    acc_rand = [100 * float(random[m]["accuracy_delta_vs_clean"]) for m in present]
    acc_dummy = [100 * float(dummy[m]["accuracy_delta_vs_clean"]) for m in present]
    axes[0].bar(x - width, acc_dummy, width, color=dummy_color, edgecolor="white", linewidth=0.8, label="Blank slots", zorder=4)
    axes[0].bar(x, acc_rand, width, color=random_color, edgecolor="white", linewidth=0.8, label="Random neurons", zorder=4)
    axes[0].bar(x + width, acc_sel, width, color=selected_color, edgecolor="white", linewidth=0.8, label="Sink neurons", zorder=4)
    axes[0].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.75, zorder=2)
    axes[0].set_title("Accuracy Drop", fontsize=20, pad=10)
    axes[0].set_ylabel("Delta vs. clean (pp)", fontsize=15)

    sink_mass = [float(selected[m].get("dummy_attention_mean", np.nan)) for m in present]
    rand_mass = [float(random[m].get("dummy_attention_mean", np.nan)) for m in present]
    axes[1].bar(x - width / 2, rand_mass, width, color=random_color, edgecolor="white", linewidth=0.8, label="Random neurons", zorder=4)
    axes[1].bar(x + width / 2, sink_mass, width, color=selected_color, edgecolor="white", linewidth=0.8, label="Sink neurons", zorder=4)
    axes[1].set_title("Inserted-Token Attention", fontsize=20, pad=10)
    axes[1].set_ylabel("Mean attention", fontsize=15)

    ctw_sel = [float(flip_selected[m]["clean_correct_to_wrong"]) / max(1.0, float(clean[m]["num_examples"])) * 100 for m in present]
    ctw_rand = [float(flip_random[m]["clean_correct_to_wrong"]) / max(1.0, float(clean[m]["num_examples"])) * 100 for m in present]
    margin_sel = [float(selected[m]["answer_margin_delta_vs_clean"]) for m in present]
    axes[2].bar(x - width / 2, ctw_rand, width, color=random_color, edgecolor="white", linewidth=0.8, label="Random neurons", zorder=4)
    axes[2].bar(x + width / 2, ctw_sel, width, color=selected_color, edgecolor="white", linewidth=0.8, label="Sink neurons", zorder=4)
    axes[2].set_title("Correct-to-Wrong Flips", fontsize=20, pad=10)
    axes[2].set_ylabel("Share of validation set (%)", fontsize=15)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=True, fancybox=False, edgecolor="#9ca3af", facecolor="white", framealpha=0.96, fontsize=13)
    fig.suptitle(args.title, fontsize=24, y=0.94, fontweight=200)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.stem}.png", dpi=300)
    fig.savefig(out / f"{args.stem}.pdf")
    plt.close(fig)
    print((out / f"{args.stem}.png").relative_to(ROOT))
    print((out / f"{args.stem}.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
