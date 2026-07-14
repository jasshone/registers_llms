from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "realtext_toy_figures"

RUNS = {
    "GPT/Pythia-style": ROOT / "outputs" / "realtext_toy_gradients" / "pythia_gpt_style_10k_posgrad16_avg8" / "online_gradient_timeline.csv",
    "Llama-style": ROOT / "outputs" / "realtext_toy_arch_gradients" / "llama_rope_10k_posgrad16_avg8" / "online_gradient_timeline.csv",
    "Qwen-style": ROOT / "outputs" / "realtext_toy_arch_gradients" / "qwen_rope_10k_posgrad16_avg8" / "online_gradient_timeline.csv",
}
CLEAN_CAUSAL = ROOT / "outputs" / "realtext_toy_sink_training" / "wikitext_pythia_tokenizer_10k_eval128" / "realtext_toy_sink_metrics.csv"
SUPPRESSED_CAUSAL = ROOT / "outputs" / "realtext_toy_causal" / "layer0_bos_key_suppressed_10k" / "realtext_toy_sink_metrics.csv"

COLORS = {
    "GPT/Pythia-style": "#3b528b",
    "Llama-style": "#21918c",
    "Qwen-style": "#5ec962",
    "Baseline": "#3b528b",
    "BOS-key ablated": "#d1495b",
}


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / "fonts"
    font_path = font_dir / "HelveticaNeueLight.otf"
    bold_path = font_dir / "HelveticaNeueBold.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.weight"] = 200
        plt.rcParams["axes.labelweight"] = 200
        plt.rcParams["axes.titleweight"] = 200
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "DejaVu Sans"
    plt.rcParams["mathtext.it"] = "DejaVu Sans:italic"
    plt.rcParams["mathtext.cal"] = "cmsy10"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def clean_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis="y", which="major", labelsize=15, width=1.4, length=5)
    ax.tick_params(axis="x", which="major", labelsize=15, width=1.4, length=5, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    if y_grid:
        ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.16, zorder=0)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                converted[key] = value
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        out.append(converted)
    return out


def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def plot_architecture_timeline() -> None:
    fig, (ax_attn, ax_loss) = plt.subplots(1, 2, figsize=(15.2, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.25, top=0.84, wspace=0.30)

    for label, path in RUNS.items():
        rows = read_csv(path)
        x = arr(rows, "step") / 1000.0
        color = COLORS[label]
        linestyle = "-"
        alpha = 1.0
        ax_attn.plot(
            x,
            arr(rows, "max_head_bos_attention"),
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            marker="o",
            markersize=4.0,
            alpha=alpha,
            label=label,
            zorder=4,
        )
        ax_loss.plot(
            x,
            arr(rows, "eval_loss"),
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            marker="o",
            markersize=4.0,
            alpha=alpha,
            label=label,
            zorder=4,
        )

    ax_attn.set_title("Attention Sink Emergence", fontsize=19, pad=8, fontweight=200)
    ax_attn.set_xlabel("Step (k)", fontsize=16, fontweight=200, labelpad=7)
    ax_attn.set_ylabel("BOS attention", fontsize=17, fontweight=200, labelpad=7)
    ax_attn.set_ylim(0.0, 0.245)
    ax_attn.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax_attn)

    ax_loss.set_title("Val. Loss", fontsize=19, pad=8, fontweight=200)
    ax_loss.set_xlabel("Step (k)", fontsize=18, fontweight=200, labelpad=8)
    ax_loss.set_ylabel("Loss", fontsize=17, fontweight=200)
    ax_loss.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax_loss)

    handles, labels = ax_attn.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#9ca3af",
        facecolor="white",
        framealpha=0.96,
        fontsize=14,
        handlelength=1.8,
        columnspacing=1.2,
    )
    save(fig, "toy_architecture_sink_timeline")


def plot_gradient_pressure() -> None:
    fig, (ax_pressure, ax_profile) = plt.subplots(1, 2, figsize=(15.2, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.95, bottom=0.25, top=0.84, wspace=0.34)

    for label, path in RUNS.items():
        rows = read_csv(path)
        x = arr(rows, "step") / 1000.0
        bos = arr(rows, "bos_logit_pressure_mean")
        nonbos = arr(rows, "nonbos_firstN_pressure_mean")
        color = COLORS[label]
        linestyle = "-"
        alpha = 1.0
        ax_pressure.plot(
            x,
            bos - nonbos,
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            marker="o",
            markersize=4.0,
            alpha=alpha,
            label=label,
            zorder=4,
        )

    gpt_rows = read_csv(RUNS["GPT/Pythia-style"])
    selected_steps = [0, 2000, 4000, 6000, 10000]
    heat_rows = []
    for step in selected_steps:
        row = next(r for r in gpt_rows if int(r["step"]) == step)
        heat_rows.append([float(row[f"keypos_pressure_{i}"]) for i in range(16)])
    heat = np.array(heat_rows, dtype=float)
    vmax = max(abs(float(np.nanmin(heat))), abs(float(np.nanmax(heat))))
    im = ax_profile.imshow(heat, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax, zorder=4)
    ax_profile.set_xticks(np.arange(16))
    ax_profile.set_yticks(np.arange(len(selected_steps)))
    ax_profile.set_yticklabels([str(s // 1000) for s in selected_steps])
    ax_profile.set_xlabel("Key pos.", fontsize=16, fontweight=200, labelpad=7)
    ax_profile.set_ylabel("Step", fontsize=13, fontweight=200, labelpad=2)
    ax_profile.set_title("Key positions", fontsize=14, pad=5, fontweight=200)
    clean_axis(ax_profile, y_grid=False)
    cbar = fig.colorbar(im, ax=ax_profile, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=12, width=1.0, length=4)

    ax_pressure.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)
    ax_pressure.set_title("Attention-logit grad.", fontsize=14, pad=5, fontweight=200)
    ax_pressure.set_xlabel("Step (k)", fontsize=16, fontweight=200, labelpad=7)
    ax_pressure.set_ylabel("BOS - other", fontsize=11, fontweight=200, labelpad=1)
    ax_pressure.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax_pressure)

    handles, labels = ax_pressure.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.34, 0.045),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#9ca3af",
        facecolor="white",
        framealpha=0.96,
        fontsize=14,
        handlelength=1.8,
        columnspacing=1.2,
    )
    save(fig, "toy_online_gradient_pressure")


def plot_causal_counterfactual() -> None:
    clean_rows = read_csv(CLEAN_CAUSAL)
    supp_rows = read_csv(SUPPRESSED_CAUSAL)
    fig, (ax_attn, ax_ret) = plt.subplots(1, 2, figsize=(15.2, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.95, bottom=0.25, top=0.84, wspace=0.34)

    for label, rows in [("Baseline", clean_rows), ("BOS-key ablated", supp_rows)]:
        x = arr(rows, "step") / 1000.0
        ax_attn.plot(
            x,
            arr(rows, "max_head_bos_attention"),
            color=COLORS[label],
            linewidth=2.5,
            marker="o",
            markersize=4.2,
            label=label,
            zorder=4,
        )
        ax_ret.plot(
            x,
            arr(rows, "zero_max_head_bos_retention"),
            color=COLORS[label],
            linewidth=2.5,
            marker="o",
            markersize=4.2,
            label=label,
            zorder=4,
        )

    ax_ret.plot(
        arr(clean_rows, "step") / 1000.0,
        arr(clean_rows, "random_zero_max_head_bos_retention"),
        color="#3b528b",
        linewidth=1.8,
        linestyle=":",
        alpha=0.75,
        label="Random control",
        zorder=3,
    )
    ax_ret.plot(
        arr(supp_rows, "step") / 1000.0,
        arr(supp_rows, "random_zero_max_head_bos_retention"),
        color="#d1495b",
        linewidth=1.8,
        linestyle=":",
        alpha=0.75,
        label="Random ablated",
        zorder=3,
    )
    ax_ret.axhline(1.0, color="#6b7280", linewidth=1.0, alpha=0.65, zorder=2)

    ax_attn.set_title("BOS attention", fontsize=14, pad=5, fontweight=200)
    ax_attn.set_xlabel("Step (k)", fontsize=16, fontweight=200, labelpad=7)
    ax_attn.set_ylabel("Attention", fontsize=11, fontweight=200, labelpad=1)
    ax_attn.set_ylim(0.0, 0.23)
    ax_attn.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax_attn)

    ax_ret.set_title("Ablation retention", fontsize=14, pad=5, fontweight=200)
    ax_ret.set_xlabel("Step (k)", fontsize=16, fontweight=200, labelpad=7)
    ax_ret.set_ylabel("Retention", fontsize=11, fontweight=200, labelpad=1)
    ax_ret.set_ylim(0.65, 1.08)
    ax_ret.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax_ret)

    handles, labels = [], []
    for ax in (ax_attn, ax_ret):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        unique.setdefault(l, h)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#9ca3af",
        facecolor="white",
        framealpha=0.96,
        fontsize=12,
        handlelength=1.6,
        columnspacing=0.85,
    )
    save(fig, "toy_causal_counterfactual")


def write_manifest() -> None:
    rows = [
        ["toy_architecture_sink_timeline", "Attention sink emergence and validation loss across architectures", "toy_architecture_sink_timeline.png", "toy_architecture_sink_timeline.pdf"],
        ["toy_online_gradient_pressure", "BOS attention-logit gradient pressure and first-16 key-position profile", "toy_online_gradient_pressure.png", "toy_online_gradient_pressure.pdf"],
        ["toy_causal_counterfactual", "Baseline versus BOS-key-suppressed training and neuron-ablation retention", "toy_causal_counterfactual.png", "toy_causal_counterfactual.pdf"],
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "description", "png", "pdf"])
        writer.writerows(rows)


def main() -> None:
    setup_style()
    missing = [path for path in list(RUNS.values()) + [CLEAN_CAUSAL, SUPPRESSED_CAUSAL] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required toy artifacts:\n" + "\n".join(str(p) for p in missing))
    plot_architecture_timeline()
    plot_gradient_pressure()
    plot_causal_counterfactual()
    write_manifest()
    print(OUT.relative_to(ROOT))
    for path in sorted(OUT.glob("*.png")):
        print(path.relative_to(ROOT))
    for path in sorted(OUT.glob("*.pdf")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
