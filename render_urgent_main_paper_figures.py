from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("outputs/preserved_main_paper_figures")
SRC_DIR = OUT_DIR / "sources"
TABLE_DIR = OUT_DIR / "tables"


SOURCES = {
    "figure2": Path(
        "outputs/realtext_toy_shifted_bos_gradients/"
        "wikitext103_train_expanded_3seed/aggregate_gradient_summary.csv"
    ),
    "figure2_cached_reference": Path(
        "outputs/realtext_toy_bos_train_position/"
        "wikitext_pythia_cached_val_windows/bos_train_position_summary.csv"
    ),
    "figure3": Path(
        "outputs/realtext_toy_bos_presence_controls/"
        "wikitext103_train_3seed/aggregate_summary.csv"
    ),
    "figure7": Path(
        "outputs/toy_pretrained_bridge/"
        "wikitext_pythia_tokenizer_10k_scale10_test512/"
        "toy_relocation_bridge_metrics.csv"
    ),
    "figure8": Path(
        "outputs/sink_hijacking_attack/repeated_bos_prefix/"
        "pythia_1b_full/repeated_bos_prefix_metrics.csv"
    ),
    "figure9_wikitext": Path("outputs/final_split_bias_transfer_source_abs_bonus0/final_results.csv"),
    "figure9_pile": Path("outputs/pile_val_sharded_transfer_full/summary_shards.csv"),
    "figure9_mmlu": Path("outputs/mmlu_transfer_eval_full_test/summary.csv"),
}


COLORS = {
    "pos0": "#2f6f9f",
    "bos": "#9f4f2f",
    "selected": "#1b7837",
    "random": "#762a83",
    "dummy": "#8c6d31",
    "clean": "#4d4d4d",
    "intervened": "#d95f02",
}


MODEL_ORDER = [
    "gpt2",
    "gpt2_medium",
    "pythia_410m",
    "pythia_2_8b",
    "llama3_2_1b",
    "qwen3_1_7b",
    "mistral_7b_v0_1",
    "phi_2",
]


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, src in SOURCES.items():
        if src.exists():
            shutil.copy2(src, SRC_DIR / f"{key}_{src.name}")


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value).replace("|", "\|")


def write_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in df.columns) + " |")
    (TABLE_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig2_bos_position_sweep() -> None:
    raw = pd.read_csv(SOURCES["figure2"]).copy()
    raw = raw[raw["variant"].astype(str).str.startswith("single_")].copy()
    raw["bos_position"] = raw["bos_positions"].astype(int)
    raw = raw.sort_values("bos_position")
    df = pd.DataFrame(
        {
            "bos_position": raw["bos_position"],
            "n_seeds": raw["n_seeds"],
            "seeds": raw["seeds"],
            "eval_loss_mean": raw["eval_loss_mean"],
            "eval_loss_std": raw["eval_loss_std"],
            "pos0_attention_mean": raw["pos0_attention_mean_mean"],
            "pos0_attention_std": raw["pos0_attention_mean_std"],
            "bos_position_attention_mean": raw["bos_attention_sum_mean_mean"],
            "bos_position_attention_std": raw["bos_attention_sum_mean_std"],
        }
    )
    write_table(df, "figure2_bos_position_sweep_exact")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.errorbar(
        df["bos_position"],
        df["pos0_attention_mean"],
        yerr=df["pos0_attention_std"],
        marker="o",
        linewidth=2,
        capsize=3,
        color=COLORS["pos0"],
        label="Absolute position 0",
    )
    ax.errorbar(
        df["bos_position"],
        df["bos_position_attention_mean"],
        yerr=df["bos_position_attention_std"],
        marker="s",
        linewidth=2,
        capsize=3,
        color=COLORS["bos"],
        label="Actual BOS position",
    )
    ax.set_xticks(df["bos_position"].tolist())
    ax.set_xlabel("BOS training position")
    ax.set_ylabel("Mean attention received")
    ax.set_title("Figure 2. BOS position sweep in toy training")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, "figure2_bos_position_sweep")


def fig3_bos_identity_controls() -> None:
    order = ["single_0", "no_bos", "single_8", "single_32", "duplicate_0,8", "duplicate_0,32"]
    df = pd.read_csv(SOURCES["figure3"])
    df["variant"] = pd.Categorical(df["variant"], order, ordered=True)
    df = df.sort_values("variant")
    write_table(
        df[
            [
                "variant",
                "bos_positions",
                "seeds",
                "n_seeds",
                "loss_mean",
                "loss_std",
                "ppl_mean",
                "ppl_std",
                "pos0_attention_mean_mean",
                "pos0_attention_mean_std",
                "bos_attention_sum_mean_mean",
                "bos_attention_sum_mean_std",
                "bos_attention_max_position_mean_mean",
                "bos_attention_max_position_mean_std",
            ]
        ],
        "figure3_bos_identity_controls_exact",
    )

    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(
        x - width / 2,
        df["pos0_attention_mean_mean"],
        width,
        yerr=df["pos0_attention_mean_std"].fillna(0),
        capsize=3,
        color=COLORS["pos0"],
        label="Position 0 attention",
    )
    ax.bar(
        x + width / 2,
        df["bos_attention_sum_mean_mean"],
        width,
        yerr=df["bos_attention_sum_mean_std"].fillna(0),
        capsize=3,
        color=COLORS["bos"],
        label="Total BOS attention",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Mean attention received")
    ax.set_title("Figure 3. BOS identity controls")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, "figure3_bos_identity_controls")


def fig7_toy_pretrained_bridge() -> None:
    df = pd.read_csv(SOURCES["figure7"]).sort_values("step")
    write_table(
        df[
            [
                "step",
                "selected_dummy_minus_bos_mean",
                "random_dummy_minus_bos_mean",
                "random_dummy_minus_bos_std",
                "dummy_only_dummy_minus_bos_mean",
                "clean_mean_bos_attention",
                "selected_ppl_ratio",
                "random_ppl_ratio_mean",
                "random_ppl_ratio_std",
                "dummy_only_ppl_ratio",
            ]
        ],
        "figure7_toy_pretrained_bridge_exact",
    )
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax.plot(
        df["step"],
        df["selected_dummy_minus_bos_mean"],
        marker="o",
        linewidth=2,
        color=COLORS["selected"],
        label="Selected transfer",
    )
    ax.plot(
        df["step"],
        df["random_dummy_minus_bos_mean"],
        marker="s",
        linewidth=2,
        color=COLORS["random"],
        label="Layer-matched random",
    )
    if "random_dummy_minus_bos_std" in df:
        ax.fill_between(
            df["step"],
            df["random_dummy_minus_bos_mean"] - df["random_dummy_minus_bos_std"].fillna(0),
            df["random_dummy_minus_bos_mean"] + df["random_dummy_minus_bos_std"].fillna(0),
            color=COLORS["random"],
            alpha=0.15,
            linewidth=0,
        )
    ax.plot(
        df["step"],
        df["dummy_only_dummy_minus_bos_mean"],
        marker="^",
        linewidth=2,
        color=COLORS["dummy"],
        label="Dummy-only control",
    )
    ax.axhline(0, color="#777777", linewidth=1)
    for step, label in [(1000, "first positive"), (3000, "strongest")]:
        row = df.loc[df["step"] == step]
        if not row.empty:
            y = float(row["selected_dummy_minus_bos_mean"].iloc[0])
            ax.scatter([step], [y], s=75, facecolor="white", edgecolor=COLORS["selected"], linewidth=2, zorder=5)
            ax.annotate(label, (step, y), xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set_ylabel("Dummy minus BOS attention")
    ax.set_title("Figure 7. Toy-to-pretrained relocation bridge")
    style_axes(ax)
    ax.legend(frameon=False, ncols=3, fontsize=8)

    ax2.plot(
        df["step"],
        df["clean_mean_bos_attention"],
        marker="o",
        linewidth=2,
        color=COLORS["clean"],
        label="Clean BOS attention",
    )
    ax2.set_xlabel("Toy checkpoint step")
    ax2.set_ylabel("Clean BOS attention")
    style_axes(ax2)
    save(fig, "figure7_toy_pretrained_bridge")


def fig8_repeated_bos_prefix_attack() -> None:
    df = pd.read_csv(SOURCES["figure8"])
    write_table(
        df[
            [
                "condition",
                "count",
                "control_id",
                "scale",
                "mean_nll",
                "ppl",
                "num_windows",
                "bos_attention_mean",
                "inserted_attention_mean",
                "inserted_minus_bos_mean",
                "ppl_ratio_vs_clean",
            ]
        ],
        "figure8_repeated_bos_prefix_attack_exact",
    )
    df = df[df["condition"] != "clean"].copy()
    order = ["selected_copy_bos", "bos_repeat_only", "random_copy_bos", "ordinary_repeat_only"]
    palette = {
        "selected_copy_bos": "#1b7837",
        "bos_repeat_only": "#2166ac",
        "random_copy_bos": "#762a83",
        "ordinary_repeat_only": "#b35806",
    }
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=False)
    for condition in order:
        sub = df[df["condition"] == condition].sort_values("count")
        if sub.empty:
            continue
        color = palette[condition]
        label = condition.replace("_", " ")
        ax.plot(sub["count"], sub["inserted_attention_mean"], marker="o", linewidth=2, color=color, label=f"{label}: inserted")
        ax.plot(sub["count"], sub["bos_attention_mean"], marker="s", linewidth=1.8, linestyle="--", color=color, alpha=0.85, label=f"{label}: BOS")
        ax2.plot(sub["count"], sub["ppl_ratio_vs_clean"], marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Repeated prefix slots")
    ax.set_ylabel("Mean attention received")
    ax.set_title("Inserted slots and remaining BOS")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, ncols=1)
    ax2.axhline(1.0, color="#777777", linewidth=1)
    ax2.set_xlabel("Repeated prefix slots")
    ax2.set_ylabel("PPL ratio")
    ax2.set_title("Perplexity effect")
    style_axes(ax2)
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 8. Repeated-BOS prefix attack", y=1.03)
    save(fig, "figure8_repeated_bos_prefix_attack")


def _label(model_key: str) -> str:
    return model_key.replace("llama3_2", "llama3.2").replace("mistral_7b_v0_1", "mistral-7b").replace("_", "-")


def fig9_frozen_generalization() -> None:
    wt = pd.read_csv(SOURCES["figure9_wikitext"])
    wt = wt[wt["status"].eq("ok")].copy()
    pile = pd.read_csv(SOURCES["figure9_pile"])
    pile = pile[pile["status"].eq("ok")].copy()
    mmlu = pd.read_csv(SOURCES["figure9_mmlu"])
    mmlu = mmlu[mmlu["status"].eq("ok")].copy()
    wt_exact = wt[
        [
            "model_key",
            "model_id",
            "family",
            "baseline_ppl",
            "intervened_ppl",
            "ppl_ratio",
            "dummy_minus_bos_mean",
            "topk",
            "scale",
        ]
    ].assign(dataset="Wikitext")
    pile_exact = pile[
        [
            "model_key",
            "model_id",
            "baseline_ppl",
            "intervened_ppl",
            "ppl_ratio",
            "dummy_minus_bos_mean",
            "source_wikitext_ppl_ratio",
            "topk",
            "scale",
        ]
    ].assign(dataset="Pile")
    mmlu_exact = mmlu[
        [
            "model_key",
            "model_id",
            "clean_accuracy",
            "intervened_accuracy",
            "accuracy_delta",
            "dummy_minus_bos_mean",
            "source_wikitext_ppl_ratio",
            "topk",
            "scale",
        ]
    ].assign(dataset="MMLU")
    write_table(wt_exact, "figure9_wikitext_exact")
    write_table(pile_exact, "figure9_pile_exact")
    write_table(mmlu_exact, "figure9_mmlu_exact")

    wt_models = set(wt["model_key"])
    pile_models = set(pile["model_key"])
    mmlu_models = set(mmlu["model_key"])
    common = [m for m in MODEL_ORDER if m in (wt_models | pile_models | mmlu_models)]

    wt_plot = wt[wt["model_key"].isin(common)][["model_key", "baseline_ppl", "intervened_ppl"]].assign(dataset="Wikitext")
    pile_plot = pile[pile["model_key"].isin(common)][["model_key", "baseline_ppl", "intervened_ppl"]].assign(dataset="Pile")
    ppl = pd.concat([wt_plot, pile_plot], ignore_index=True)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.4), gridspec_kw={"height_ratios": [1.45, 1.0]})

    datasets = ["Wikitext", "Pile"]
    x_labels = []
    x = []
    clean_vals = []
    int_vals = []
    pos = 0
    for model in common:
        for dataset in datasets:
            sub = ppl[(ppl["model_key"] == model) & (ppl["dataset"] == dataset)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            x.append(pos)
            x_labels.append(f"{_label(model)}\n{dataset}")
            clean_vals.append(row["baseline_ppl"])
            int_vals.append(row["intervened_ppl"])
            pos += 1
        pos += 0.7
    x = np.array(x, dtype=float)
    width = 0.36
    ax.bar(x - width / 2, clean_vals, width, color=COLORS["clean"], label="Clean")
    ax.bar(x + width / 2, int_vals, width, color=COLORS["intervened"], label="Intervention")
    ax.set_yscale("log")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Wikitext and Pile")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    style_axes(ax)
    ax.legend(frameon=False, ncols=2)

    mmlu = mmlu[mmlu["model_key"].isin(common)].copy()
    mmlu["model_key"] = pd.Categorical(mmlu["model_key"], common, ordered=True)
    mmlu = mmlu.sort_values("model_key")
    x2 = np.arange(len(mmlu))
    ax2.bar(x2 - width / 2, mmlu["clean_accuracy"], width, color=COLORS["clean"], label="Clean")
    ax2.bar(x2 + width / 2, mmlu["intervened_accuracy"], width, color=COLORS["intervened"], label="Intervention")
    ax2.set_ylim(0, max(0.65, float(mmlu[["clean_accuracy", "intervened_accuracy"]].max().max()) * 1.15))
    ax2.set_ylabel("MMLU accuracy")
    ax2.set_title("MMLU test")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([_label(m) for m in mmlu["model_key"].astype(str)], rotation=30, ha="right", fontsize=8)
    style_axes(ax2)
    ax2.legend(frameon=False, ncols=2)
    fig.suptitle("Figure 9. Frozen-selection generalization", y=1.02)
    save(fig, "figure9_frozen_selection_generalization")


def write_manifest() -> None:
    lines = [
        "# Preserved Main-Paper Figures",
        "",
        "Rendered from existing result files; no training or exhaustive grid was run.",
        "",
        "## Figures",
        "- Figure 2: `figure2_bos_position_sweep.{png,pdf}`",
        "- Figure 3: `figure3_bos_identity_controls.{png,pdf}`",
        "- Figure 7: `figure7_toy_pretrained_bridge.{png,pdf}`",
        "- Figure 8: `figure8_repeated_bos_prefix_attack.{png,pdf}`",
        "- Figure 9: `figure9_frozen_selection_generalization.{png,pdf}`",
        "",
        "## Exact Number Tables",
        "- Figure 2: `tables/figure2_bos_position_sweep_exact.{csv,md}`",
        "- Figure 3: `tables/figure3_bos_identity_controls_exact.{csv,md}`",
        "- Figure 7: `tables/figure7_toy_pretrained_bridge_exact.{csv,md}`",
        "- Figure 8: `tables/figure8_repeated_bos_prefix_attack_exact.{csv,md}`",
        "- Figure 9 Wikitext: `tables/figure9_wikitext_exact.{csv,md}`",
        "- Figure 9 Pile: `tables/figure9_pile_exact.{csv,md}`",
        "- Figure 9 MMLU: `tables/figure9_mmlu_exact.{csv,md}`",
        "",
        "## Source CSVs",
    ]
    for key, src in SOURCES.items():
        lines.append(f"- {key}: `{src}`")
    lines.extend(
        [
            "",
            "Notes:",
            "- Figure 2 now uses the train-text Wikitext103 expanded 3-seed shifted-BOS aggregate. The older cached validation-window sweep is copied as `figure2_cached_reference_bos_train_position_summary.csv`.",
            "- Figure 3 uses the available aggregate seed summary; variants with a single seed have zero/blank visible error bars.",
        ]
    )
    (OUT_DIR / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup()
    fig2_bos_position_sweep()
    fig3_bos_identity_controls()
    fig7_toy_pretrained_bridge()
    fig8_repeated_bos_prefix_attack()
    fig9_frozen_generalization()
    write_manifest()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
