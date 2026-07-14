from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "toy_pretrained_bridge" / "final_method_diagnosis"
BOS_DIR = ROOT / "outputs" / "toy_pretrained_bridge" / "final_method_bos_allckpt_smallgrid"
PIPE_DIR = ROOT / "outputs" / "toy_pretrained_bridge" / "final_method_pipeline_allckpt_smallgrid"
OFFICIAL = ROOT / "outputs" / "final_split_bias_transfer_official_test_source_abs_bonus0" / "official_test_results.csv"


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
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def clean_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis="y", which="major", labelsize=12, width=1.2, length=5)
    ax.tick_params(axis="x", which="major", labelsize=12, width=1.2, length=5, pad=7)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.3)
    if y_grid:
        ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.15, zorder=0)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        cur: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                cur[key] = value
                continue
            try:
                cur[key] = float(value)
            except ValueError:
                cur[key] = value
        out.append(cur)
    return out


def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def best_val_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    transfer = [r for r in rows if r.get("mode") == "transfer" and r.get("status") == "ok"]
    best: dict[int, dict[str, Any]] = {}
    for row in transfer:
        step = int(row["step"])
        prev = best.get(step)
        if prev is None or float(row["dummy_minus_bos_mean"]) > float(prev["dummy_minus_bos_mean"]):
            best[step] = row
    return best


def official_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transfers = [r for r in rows if r.get("mode") == "transfer" and r.get("status") == "ok"]
    positive = [r for r in transfers if float(r["dummy_minus_bos_mean"]) > 0.0]
    pool = positive or transfers
    strong = [r for r in pool if float(r["dummy_minus_bos_mean"]) >= 0.30]
    if strong:
        return min(strong, key=lambda r: (float(r["ppl_ratio"]), int(r["topk"]), -float(r["dummy_minus_bos_mean"])))
    if positive:
        return min(positive, key=lambda r: (float(r["ppl_ratio"]), int(r["topk"]), -float(r["dummy_minus_bos_mean"])))
    return max(transfers, key=lambda r: float(r["dummy_minus_bos_mean"]))


def official_pythia_1b() -> dict[str, Any]:
    for row in read_rows(OFFICIAL):
        if row.get("model_key") == "pythia_1b":
            return row
    raise RuntimeError("missing pythia_1b official result")


def write_combined_summary(
    bos_test: list[dict[str, Any]],
    pipe_test: list[dict[str, Any]],
    bos_val: list[dict[str, Any]],
    pipe_val: list[dict[str, Any]],
    pythia: dict[str, Any],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "bos_selected_reloc",
        "pipeline_selected_reloc",
        "official_selected_reloc",
        "official_selected_mask",
        "best_validation_reloc",
        "best_validation_mask",
        "toy_bos_before_mean",
        "toy_train_max_head_bos",
        "selected_ppl_ratio",
    ]
    combined_rows: list[dict[str, Any]] = []
    bos_by_step = {int(r["step"]): r for r in bos_test}
    pipe_by_step = {int(r["step"]): r for r in pipe_test}
    bos_best = best_val_rows(bos_val)
    pipe_best = best_val_rows(pipe_val)
    val_by_step: dict[int, list[dict[str, Any]]] = {}
    for row in bos_val + pipe_val:
        val_by_step.setdefault(int(row["step"]), []).append(row)
    for step in sorted(set(bos_by_step) | set(pipe_by_step)):
        selected_val = official_choice(val_by_step[step])
        candidates = [
            r
            for r in (bos_by_step.get(step), pipe_by_step.get(step))
            if r is not None
            and r["mask"] == selected_val["mask"]
            and int(r["topk"]) == int(selected_val["topk"])
            and abs(float(r["scale"]) - float(selected_val["scale"])) < 1e-9
        ]
        if not candidates:
            raise RuntimeError(f"missing test row for selected validation row at step {step}: {selected_val}")
        selected = candidates[0]
        val_candidates = [r for r in (bos_best.get(step), pipe_best.get(step)) if r is not None]
        best_val = max(val_candidates, key=lambda r: float(r["dummy_minus_bos_mean"]))
        context_path = BOS_DIR / f"context_step{step}.json"
        max_head = ""
        if context_path.exists():
            import json

            max_head = json.loads(context_path.read_text()).get("train_max_head_bos", "")
        combined_rows.append(
            {
                "step": step,
                "bos_selected_reloc": bos_by_step.get(step, {}).get("dummy_minus_bos_mean", ""),
                "pipeline_selected_reloc": pipe_by_step.get(step, {}).get("dummy_minus_bos_mean", ""),
                "official_selected_reloc": selected["dummy_minus_bos_mean"],
                "official_selected_mask": selected["mask"],
                "best_validation_reloc": best_val["dummy_minus_bos_mean"],
                "best_validation_mask": best_val["mask"],
                "toy_bos_before_mean": selected["bos_before_mean"],
                "toy_train_max_head_bos": max_head,
                "selected_ppl_ratio": selected["ppl_ratio"],
            }
        )
    with (OUT_DIR / "toy_final_method_combined_diagnosis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined_rows)

    final = combined_rows[-1]
    max_combined = max(combined_rows, key=lambda r: float(r["official_selected_reloc"]))
    max_best = max(combined_rows, key=lambda r: float(r["best_validation_reloc"]))
    lines = [
        "# Toy Final-Method Diagnosis",
        "",
        "This combines the toy `bos_only` and `pipeline_sinks_plus_bos` reruns using the final fixed-bias transfer protocol from the paper figures.",
        "",
        "## Key numbers",
        "",
        f"- Final checkpoint official-selected relocation: `{float(final['official_selected_reloc']):.4g}` using `{final['official_selected_mask']}`.",
        f"- Strongest official-selected test relocation: step `{int(max_combined['step'])}`, `{float(max_combined['official_selected_reloc']):.4g}`.",
        f"- Strongest validation relocation anywhere in the grid: step `{int(max_best['step'])}`, `{float(max_best['best_validation_reloc']):.4g}` using `{max_best['best_validation_mask']}`.",
        f"- Toy final BOS-before mean: `{float(final['toy_bos_before_mean']):.4g}`.",
        f"- Toy final max-head BOS attention: `{float(final['toy_train_max_head_bos']):.4g}`.",
        f"- Official Pythia-1B BOS-before mean: `{float(pythia['bos_before_mean']):.4g}`.",
        f"- Official Pythia-1B dummy-minus-BOS: `{float(pythia['dummy_minus_bos_mean']):.4g}`.",
        "",
        "## Interpretation",
        "",
        "The toy does show final-method relocation, but it is much smaller than pretrained Pythia-1B. The main reason is not a missing scale sweep: high-scale rows exist and can move more attention. The gap comes from the toy sink being weaker and less distributed across layers/heads, plus the official validation selector preferring low-PPL positive rows over high-relocation rows.",
        "",
        "Mean BOS attention is low partly because it averages over every layer/head/query. The final toy has a max-head BOS value around 0.21 in earlier bridge diagnostics, but the all-layer mean here is only about 0.073. Official Pythia-1B has an all-layer mean around 0.34, so the pretrained sink is genuinely broader and stronger.",
    ]
    (OUT_DIR / "toy_final_method_diagnosis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bos_test = read_rows(BOS_DIR / "test_selected_rows.csv")
    pipe_test = read_rows(PIPE_DIR / "test_selected_rows.csv")
    bos_val = read_rows(BOS_DIR / "validation_grid.csv")
    pipe_val = read_rows(PIPE_DIR / "validation_grid.csv")
    pythia = official_pythia_1b()
    write_combined_summary(bos_test, pipe_test, bos_val, pipe_val, pythia)

    combined = read_rows(OUT_DIR / "toy_final_method_combined_diagnosis.csv")
    x = arr(combined, "step") / 1000.0
    colors = {
        "bos": "#3b528b",
        "pipe": "#21918c",
        "best": "#d1495b",
        "ref": "#6b7280",
        "max": "#5ec962",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.22, wspace=0.34)

    ax = axes[0]
    ax.axhline(0.0, color=colors["ref"], linewidth=1.0, alpha=0.7, zorder=1)
    ax.plot(x, arr(bos_test, "dummy_minus_bos_mean"), color=colors["bos"], marker="o", linewidth=2.2, markersize=4.0, label="BOS source")
    ax.plot(x, arr(pipe_test, "dummy_minus_bos_mean"), color=colors["pipe"], marker="o", linewidth=2.4, markersize=4.0, label="Sink+BOS source")
    ax.set_title("Fixed-Bias Relocation", fontsize=17, pad=8, fontweight=200)
    ax.set_xlabel("Step (k)", fontsize=13, labelpad=6)
    ax.set_ylabel("Dummy - BOS attention", fontsize=12, labelpad=6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.5)

    ax = axes[1]
    ax.axhline(0.0, color=colors["ref"], linewidth=1.0, alpha=0.7, zorder=1)
    ax.plot(x, arr(combined, "official_selected_reloc"), color=colors["pipe"], marker="o", linewidth=2.4, markersize=4.0, label="Selected test")
    ax.plot(x, arr(combined, "best_validation_reloc"), color=colors["best"], marker="s", linewidth=2.0, markersize=3.8, label="Best validation")
    ax.set_title("Selection Effect", fontsize=17, pad=8, fontweight=200)
    ax.set_xlabel("Step (k)", fontsize=13, labelpad=6)
    ax.set_ylabel("Relocation", fontsize=12, labelpad=6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.5)

    ax = axes[2]
    ax.plot(x, arr(combined, "toy_bos_before_mean"), color=colors["pipe"], marker="o", linewidth=2.4, markersize=4.0, label="Toy mean")
    ax.plot(x, arr(combined, "toy_train_max_head_bos"), color=colors["max"], marker="s", linewidth=2.0, markersize=3.8, label="Toy max head")
    ax.axhline(float(pythia["bos_before_mean"]), color=colors["ref"], linestyle="--", linewidth=1.5, label="Pythia-1B mean")
    ax.set_title("BOS Attention Scale", fontsize=17, pad=8, fontweight=200)
    ax.set_xlabel("Step (k)", fontsize=13, labelpad=6)
    ax.set_ylabel("BOS attention", fontsize=12, labelpad=6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.5)

    fig.savefig(OUT_DIR / "toy_final_method_diagnosis.png", dpi=300)
    fig.savefig(OUT_DIR / "toy_final_method_diagnosis.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
