from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "pythia_bos_logit_fd"
OUT = ROOT / "outputs" / "pythia_bos_logit_fd"
MODELS = [
    ("pythia_70m", "70M", "32w_256tok_eps0.05"),
    ("pythia_160m", "160M", "16w_256tok_eps0.05"),
    ("pythia_1b", "1B", "8w_256tok_eps0.05"),
]


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / "fonts"
    font_path = font_dir / "HelveticaNeueLight.otf"
    bold_path = font_dir / "HelveticaNeueBold.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.weight"] = 300
        plt.rcParams["axes.labelweight"] = 300
        plt.rcParams["axes.titleweight"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        parsed: dict[str, Any] = dict(row)
        for key, val in row.items():
            try:
                parsed[key] = float(val)
            except (TypeError, ValueError):
                pass
        out.append(parsed)
    return out


def by_control(rows: list[dict[str, Any]], control: str) -> list[dict[str, Any]]:
    return sorted([r for r in rows if r["control"] == control], key=lambda r: float(r["step"]))


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#9ca3af", alpha=0.18, linewidth=1.0)
    ax.tick_params(axis="both", labelsize=12, width=1.2, length=4)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)


def plot_model(ax_attn: plt.Axes, ax_deriv: plt.Axes, rows: list[dict[str, Any]], label: str) -> None:
    colors = {"bos": "#d1495b", "pos5": "#3b528b"}
    names = {"bos": "BOS", "pos5": "pos5 control"}
    for control in ("bos", "pos5"):
        cur = by_control(rows, control)
        x = np.array([float(r["step"]) for r in cur])
        attn = np.array([float(r["mean_attention"]) for r in cur])
        min_deriv = np.array([float(r["min_layer_derivative"]) for r in cur])
        ax_attn.plot(x, attn, marker="o", linewidth=2.1, color=colors[control], label=names[control])
        ax_deriv.plot(x, min_deriv, marker="o", linewidth=2.1, color=colors[control], label=names[control])
    ax_deriv.axhline(0.0, color="#4b5563", linewidth=1.0, alpha=0.75)
    ax_attn.set_title(label, fontsize=17, pad=8)
    ax_attn.set_ylabel("Mean attention", fontsize=13)
    ax_deriv.set_ylabel("Direct Effect", fontsize=13)
    ax_deriv.set_xlabel("Training step", fontsize=13)
    for ax in (ax_attn, ax_deriv):
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_xticks([0, 1000, 4000, 16000, 64000, 143000])
        ax.set_xticklabels(["0", "1k", "4k", "16k", "64k", "143k"])
        clean_axis(ax)
    ax_attn.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax_deriv.yaxis.set_major_locator(ticker.MaxNLocator(4))



def save_padded_figure(fig: plt.Figure, out_png: Path, out_pdf: Path, pad: int = 100) -> None:
    tmp_png = out_png.with_name(out_png.stem + "_raw_tmp.png")
    fig.savefig(tmp_png, dpi=300)
    img = Image.open(tmp_png).convert("RGB")
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), "white")
    canvas.paste(img, (pad, pad))
    canvas.save(out_png)
    tmp_png.unlink(missing_ok=True)
    pad_fig, ax = plt.subplots(figsize=(canvas.width / 300, canvas.height / 300), dpi=300)
    ax.imshow(canvas)
    ax.axis("off")
    pad_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pad_fig.savefig(out_pdf, dpi=300)
    plt.close(pad_fig)

def main() -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.84, wspace=0.28, hspace=0.34)
    for idx, (model_key, label, subdir) in enumerate(MODELS):
        rows = read_rows(SRC / model_key / subdir / "bos_logit_fd.csv")
        plot_model(axes[0, idx], axes[1, idx], rows, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=2, frameon=False, fontsize=13)
    fig.suptitle("BOS Direct Effect", fontsize=24, y=0.965, fontweight=300)
    OUT.mkdir(parents=True, exist_ok=True)
    save_padded_figure(
        fig,
        OUT / "pythia_bos_logit_direct_effect.png",
        OUT / "pythia_bos_logit_direct_effect.pdf",
    )
    plt.close(fig)
    print((OUT / "pythia_bos_logit_direct_effect.png").relative_to(ROOT))
    print((OUT / "pythia_bos_logit_direct_effect.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
