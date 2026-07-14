from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "official_family_scale_figures_grouped_bars"

TRANSFER_RESULTS = ROOT / "outputs" / "final_split_bias_transfer_official_test_source_abs_bonus0" / "official_test_results.csv"
ZERO_RESULTS = ROOT / "outputs" / "final_split_bias_transfer_official_val" / "split_test_zero_source_only_results.csv"
RESCUE_TEST_RESULTS = {
    "pythia_2_8b": ROOT / "outputs" / "final_split_bias_transfer_official_test_pythia28_rescue_scale375" / "official_test_results.csv",
    "pythia_12b": ROOT / "outputs" / "final_split_bias_transfer" / "pythia_12b" / "rescue_test_result.csv",
    "pythia_160m": ROOT / "outputs" / "final_split_bias_transfer" / "pythia_160m" / "rescue_test_result.csv",
    "pythia_70m": ROOT / "outputs" / "final_split_bias_transfer" / "pythia_70m" / "rescue_test_result.csv",
    "llama3_8b": ROOT / "outputs" / "final_split_bias_transfer" / "llama3_8b" / "rescue_test_result.csv",
}

FAMILY_ORDER = {
    "pythia": [
        ("pythia_70m", "70M", 70e6),
        ("pythia_160m", "160M", 160e6),
        ("pythia_410m", "410M", 410e6),
        ("pythia_1b", "1B", 1e9),
        ("pythia_1_4b", "1.4B", 1.4e9),
        ("pythia_2_8b", "2.8B", 2.8e9),
        ("pythia_6_9b", "6.9B", 6.9e9),
        ("pythia_12b", "12B", 12e9),
    ],
    "llama": [
        ("llama3_2_1b", "1B", 1e9),
        ("llama3_2_3b", "3B", 3e9),
        ("llama3_8b", "8B", 8e9),
    ],
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


def ordered_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = []
    missing = []
    for rank, (model_key, label, params) in enumerate(FAMILY_ORDER[family]):
        hit = df[(df["family"] == family) & (df["model_key"] == model_key)]
        if hit.empty:
            missing.append(model_key)
            continue
        row = hit.iloc[0].copy()
        row["family_rank"] = rank
        row["display_label"] = label
        row["params"] = params
        rows.append(row)
    if missing:
        raise ValueError(f"Missing {family} models in result table: {', '.join(missing)}")
    return pd.DataFrame(rows).sort_values("family_rank")


def load_transfer_results() -> pd.DataFrame:
    df = pd.read_csv(TRANSFER_RESULTS).copy()
    df["row_source"] = str(TRANSFER_RESULTS.relative_to(ROOT))

    for model_key, path in RESCUE_TEST_RESULTS.items():
        if not path.exists():
            continue
        rescue = pd.read_csv(path).copy()
        rescue = rescue[rescue["model_key"] == model_key]
        if rescue.empty:
            continue
        rescue = rescue.iloc[[0]].copy()
        rescue["row_source"] = str(path.relative_to(ROOT))
        for col in df.columns:
            if col not in rescue.columns:
                rescue[col] = np.nan
        extra_cols = [col for col in rescue.columns if col not in df.columns]
        if extra_cols:
            df = df.assign(**{col: np.nan for col in extra_cols})
        df = df[df["model_key"] != model_key]
        df = pd.concat([df, rescue[df.columns]], ignore_index=True)
    return df


def clean_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis="y", which="major", labelsize=15, width=1.4, length=5)
    ax.tick_params(axis="x", which="major", labelsize=15, width=1.4, length=0, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    if y_grid:
        ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.16, zorder=0)



def plot_family(df: pd.DataFrame, family: str, kind: str, filename: str) -> pd.DataFrame:
    family_df = ordered_family(df[df["status"] == "ok"], family)
    x = np.arange(len(family_df))
    pre_color = plt.cm.viridis(0.18)
    post_color = plt.cm.viridis(0.74)

    fig, (ax_attn, ax_ppl) = plt.subplots(1, 2, figsize=(15.2, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.955, bottom=0.28, top=0.84, wspace=0.34)

    labels = family_df["display_label"].tolist()
    base_attention = family_df["bos_before_mean"].astype(float).to_numpy()
    dummy_after = family_df["dummy_after_mean"].astype(float).to_numpy()
    base_ppl = family_df["baseline_ppl"].astype(float).to_numpy()
    changed_ppl = family_df["intervened_ppl"].astype(float).to_numpy()

    bar_width = 0.34
    bar_gap = bar_width / 2

    ax_attn.bar(
        x - bar_gap,
        base_attention,
        bar_width,
        color=pre_color,
        edgecolor="white",
        linewidth=0.9,
        label="Pre-intervention attention",
        zorder=4,
    )
    ax_attn.bar(
        x + bar_gap,
        dummy_after,
        bar_width,
        color=post_color,
        edgecolor="white",
        linewidth=0.9,
        label="Post-intervention attention",
        zorder=4,
    )
    ax_attn.set_xticks(x)
    ax_attn.set_xticklabels(labels)
    ax_attn.set_ylabel("Mean Attention", fontsize=21, fontweight=200)
    ax_attn.set_xlabel("Model Scale", fontsize=21, fontweight=200, labelpad=10)
    ax_attn.set_title("Attention Redistribution", fontsize=25, pad=10, fontweight=200)
    ax_attn.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax_attn)

    ax_ppl.bar(
        x - bar_gap,
        base_ppl,
        bar_width,
        color=pre_color,
        edgecolor="white",
        linewidth=0.9,
        label="Pre-intervention PPL",
        zorder=4,
    )
    ax_ppl.bar(
        x + bar_gap,
        changed_ppl,
        bar_width,
        color=post_color,
        edgecolor="white",
        linewidth=0.9,
        label=f"{kind} PPL",
        zorder=4,
    )
    ax_ppl.set_xticks(x)
    ax_ppl.set_xticklabels(labels)
    ax_ppl.set_yscale("log")
    ax_ppl.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:g}"))
    ax_ppl.set_ylabel("Perplexity", fontsize=21, fontweight=200)
    ax_ppl.set_xlabel("Model Scale", fontsize=21, fontweight=200, labelpad=10)
    ax_ppl.set_title("Perplexity Change", fontsize=25, pad=10, fontweight=200)
    clean_axis(ax_ppl)

    handles, legend_labels = [], []
    for ax in (ax_attn, ax_ppl):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        legend_labels.extend(l)
    unique = dict(zip(legend_labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=min(4, len(unique)),
        frameon=True,
        fancybox=False,
        edgecolor="#9ca3af",
        facecolor="white",
        framealpha=0.96,
        fontsize=14,
        handlelength=1.5,
        columnspacing=1.25,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{filename}.png"
    pdf = OUT / f"{filename}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    manifest_cols = [
        "family",
        "model_key",
        "model_id",
        "split",
        "mask",
        "mode",
        "topk",
        "scale",
        "layers",
        "bos_before_mean",
        "bos_after_mean",
        "dummy_after_mean",
        "dummy_minus_bos_mean",
        "baseline_ppl",
        "intervened_ppl",
        "ppl_ratio",
        "ppl_change_pct",
        "row_source",
    ]
    return family_df[[c for c in manifest_cols if c in family_df.columns]].assign(
        figure_png=str(png.relative_to(ROOT)),
        figure_pdf=str(pdf.relative_to(ROOT)),
    )


def plot_family_panel(df: pd.DataFrame, family: str, kind: str, filename: str, panel: str) -> pd.DataFrame:
    family_df = ordered_family(df[df["status"] == "ok"], family)
    x = np.arange(len(family_df))
    pre_color = plt.cm.viridis(0.18)
    post_color = plt.cm.viridis(0.74)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.34, top=0.84)

    labels = family_df["display_label"].tolist()
    bar_width = 0.34
    bar_gap = bar_width / 2

    if panel == "attention":
        before = family_df["bos_before_mean"].astype(float).to_numpy()
        after = family_df["dummy_after_mean"].astype(float).to_numpy()
        ax.bar(
            x - bar_gap,
            before,
            bar_width,
            color=pre_color,
            edgecolor="white",
            linewidth=0.9,
            label="Pre-intervention attention",
            zorder=4,
        )
        ax.bar(
            x + bar_gap,
            after,
            bar_width,
            color=post_color,
            edgecolor="white",
            linewidth=0.9,
            label="Post-intervention attention",
            zorder=4,
        )
        ax.set_ylabel("Mean Attention", fontsize=21, fontweight=200)
        ax.set_title("Attention Redistribution", fontsize=25, pad=10, fontweight=200)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    elif panel == "ppl":
        before = family_df["baseline_ppl"].astype(float).to_numpy()
        after = family_df["intervened_ppl"].astype(float).to_numpy()
        ax.bar(
            x - bar_gap,
            before,
            bar_width,
            color=pre_color,
            edgecolor="white",
            linewidth=0.9,
            label="Pre-intervention PPL",
            zorder=4,
        )
        ax.bar(
            x + bar_gap,
            after,
            bar_width,
            color=post_color,
            edgecolor="white",
            linewidth=0.9,
            label=f"{kind} PPL",
            zorder=4,
        )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:g}"))
        ax.set_ylabel("Perplexity", fontsize=21, fontweight=200)
        ax.set_title("Perplexity Change", fontsize=25, pad=10, fontweight=200)
    else:
        raise ValueError(f"unknown panel: {panel}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model Scale", fontsize=21, fontweight=200, labelpad=10)
    clean_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.46),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#9ca3af",
        facecolor="white",
        framealpha=0.96,
        fontsize=14,
        handlelength=1.5,
        columnspacing=1.25,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{filename}.png"
    pdf = OUT / f"{filename}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    manifest_cols = [
        "family",
        "model_key",
        "model_id",
        "split",
        "mask",
        "mode",
        "topk",
        "scale",
        "layers",
        "bos_before_mean",
        "bos_after_mean",
        "dummy_after_mean",
        "dummy_minus_bos_mean",
        "baseline_ppl",
        "intervened_ppl",
        "ppl_ratio",
        "ppl_change_pct",
        "row_source",
    ]
    return family_df[[c for c in manifest_cols if c in family_df.columns]].assign(
        figure_png=str(png.relative_to(ROOT)),
        figure_pdf=str(pdf.relative_to(ROOT)),
        panel=panel,
    )


def main() -> None:
    setup_style()
    transfer = load_transfer_results()
    zero = pd.read_csv(ZERO_RESULTS)
    zero["row_source"] = str(ZERO_RESULTS.relative_to(ROOT))

    manifests = []
    for family in ("pythia", "llama"):
        manifests.append(
            plot_family(
                transfer,
                family,
                "Intervened",
                f"{family}_official_test_attention_ppl",
            ).assign(figure_type="official_test", panel="combined")
        )
        manifests.append(
            plot_family(
                zero,
                family,
                "Zeroing",
                f"{family}_zero_source_ablation_attention_ppl",
            ).assign(figure_type="zero_source_ablation", panel="combined")
        )
        if family == "pythia":
            manifests.append(
                plot_family_panel(
                    transfer,
                    family,
                    "Intervened",
                    "pythia_official_test_attention",
                    "attention",
                ).assign(figure_type="official_test_split")
            )
            manifests.append(
                plot_family_panel(
                    transfer,
                    family,
                    "Intervened",
                    "pythia_official_test_ppl",
                    "ppl",
                ).assign(figure_type="official_test_split")
            )
            manifests.append(
                plot_family_panel(
                    zero,
                    family,
                    "Zeroing",
                    "pythia_zero_source_ablation_attention",
                    "attention",
                ).assign(figure_type="zero_source_ablation_split")
            )
            manifests.append(
                plot_family_panel(
                    zero,
                    family,
                    "Zeroing",
                    "pythia_zero_source_ablation_ppl",
                    "ppl",
                ).assign(figure_type="zero_source_ablation_split")
            )

    pd.concat(manifests, ignore_index=True).to_csv(OUT / "figure_manifest.csv", index=False)
    print(f"Wrote figures and manifest to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
