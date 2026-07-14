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
DEFAULT_SRC = (
    ROOT
    / "outputs"
    / "realtext_toy_bos_position_intervention"
    / "wikitext_pythia_tokenizer_10k"
)


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
    ax.tick_params(axis="both", labelsize=11, width=1.2, length=4)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--stem", default="toy_bos_position_intervention")
    args = parser.parse_args()

    src = args.src if args.src.is_absolute() else ROOT / args.src
    rows = read_rows(src / "toy_bos_position_intervention.csv")
    steps = sorted({int(r["step"]) for r in rows})
    positions = sorted({int(r["insert_pos"]) for r in rows})
    by_key = {(int(r["step"]), int(r["insert_pos"])): r for r in rows}

    lift = np.array(
        [[100 * float(by_key[(step, pos)]["bos_inserted_attention_lift"]) for pos in positions] for step in steps]
    )

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.82, wspace=0.34)

    vmax = max(abs(float(np.nanmin(lift))), abs(float(np.nanmax(lift))))
    im = axes[0].imshow(lift, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, zorder=3)
    axes[0].set_title("BOS Attention Lift", fontsize=19, pad=10)
    axes[0].set_xlabel("Insertion position", fontsize=14)
    axes[0].set_ylabel("Training step", fontsize=14)
    axes[0].set_xticks(np.arange(len(positions)))
    axes[0].set_xticklabels([str(p) for p in positions])
    axes[0].set_yticks(np.arange(len(steps)))
    axes[0].set_yticklabels([f"{s // 1000}k" if s else "0" for s in steps])
    clean_axis(axes[0])
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.03)
    cbar.set_label("Attention lift (pp)", fontsize=12)
    cbar.ax.tick_params(labelsize=10, width=1.0, length=3)

    colors = {
        1: "#4b5563",
        4: "#3b82f6",
        8: "#21918c",
        16: "#f59e0b",
        32: "#d1495b",
        64: "#7c3aed",
    }
    shown_positions = [p for p in [1, 4, 8, 16, 32, 64] if p in positions]
    for pos in shown_positions:
        y = [float(by_key[(step, pos)]["bos_minus_control_loss"]) for step in steps]
        axes[1].plot(steps, y, marker="o", markersize=4.5, linewidth=2.0, color=colors[pos], label=f"pos {pos}")
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7, zorder=1)
    axes[1].set_title("BOS Loss Effect", fontsize=19, pad=10)
    axes[1].set_xlabel("Training step", fontsize=14)
    axes[1].set_ylabel("Loss delta", fontsize=14)
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{int(x / 1000)}k"))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(6))
    axes[1].grid(False)
    axes[1].yaxis.grid(True, color="#6b7280", linewidth=1.0, alpha=0.16, zorder=0)
    clean_axis(axes[1])
    axes[1].legend(frameon=True, fancybox=False, edgecolor="#9ca3af", fontsize=11, ncol=2)

    fig.suptitle("Toy BOS Position Intervention", fontsize=23, y=0.94, fontweight=200)
    out = src / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.stem}.png", dpi=300)
    fig.savefig(out / f"{args.stem}.pdf")
    plt.close(fig)
    print((out / f"{args.stem}.png").relative_to(ROOT))
    print((out / f"{args.stem}.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
