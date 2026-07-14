from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("outputs/bootstrap_postprocess_20260714")
DEFAULT_SAMPLES = 10_000
DEFAULT_SEED = 17

INTERVENTION_DECOMPOSITION = {
    "pythia_1b": Path("outputs/intervention_decomposition/pythia_1b_20260714_run2/per_example.csv"),
    "llama3_2_3b": Path("outputs/intervention_decomposition/llama3_2_3b_20260714/per_example.csv"),
    "qwen3_4b": Path("outputs/intervention_decomposition/qwen3_4b_20260714/per_example.csv"),
}

TOY_BRIDGE = Path("outputs/preserved_main_paper_figures/sources/figure7_toy_relocation_bridge_metrics.csv")

AGGREGATE_ONLY_SOURCES = {
    "pile": Path("outputs/preserved_main_paper_figures/sources/figure9_pile_summary_shards.csv"),
    "mmlu": Path("outputs/preserved_main_paper_figures/sources/figure9_mmlu_summary.csv"),
    "repeated_bos_pythia_1b": Path("outputs/preserved_main_paper_figures/sources/figure8_repeated_bos_prefix_metrics.csv"),
    "repeated_bos_llama3_2_3b": Path("outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714/repeated_bos_prefix_metrics.csv"),
    "repeated_bos_qwen3_4b": Path("outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714/repeated_bos_prefix_metrics.csv"),
}


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: np.ndarray, *, rng: np.random.Generator, samples: int) -> BootstrapResult:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    if clean.size == 1:
        val = float(clean[0])
        return BootstrapResult(val, val, val, 1)
    idx = rng.integers(0, clean.size, size=(int(samples), clean.size))
    means = clean[idx].mean(axis=1)
    return BootstrapResult(
        mean=float(clean.mean()),
        ci_low=float(np.quantile(means, 0.025)),
        ci_high=float(np.quantile(means, 0.975)),
        n=int(clean.size),
    )


def parse_float_list(text: Any) -> list[float]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    return [float(part) for part in str(text).replace(",", " ").split() if part]


def postprocess_intervention_decomposition(out: Path, *, rng: np.random.Generator, samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, path in INTERVENTION_DECOMPOSITION.items():
        if not path.exists():
            rows.append(
                {
                    "artifact_family": "intervention_decomposition",
                    "model_key": model_key,
                    "source_path": str(path),
                    "status": "missing_source",
                }
            )
            continue
        df = pd.read_csv(path)
        for (condition, candidate_group), group in df.groupby(["condition", "candidate_group"], dropna=False):
            delta_nll = group["delta_nll"].to_numpy(dtype=np.float64)
            delta_boot = bootstrap_mean(delta_nll, rng=rng, samples=samples)
            row: dict[str, Any] = {
                "artifact_family": "intervention_decomposition",
                "model_key": model_key,
                "condition": condition,
                "candidate_group": candidate_group,
                "source_path": str(path),
                "resampling_unit": "example_idx",
                "num_units": delta_boot.n,
                "delta_nll_mean": delta_boot.mean,
                "delta_nll_ci_low": delta_boot.ci_low,
                "delta_nll_ci_high": delta_boot.ci_high,
                "ppl_ratio_mean_from_delta_nll": math.exp(delta_boot.mean) if math.isfinite(delta_boot.mean) else float("nan"),
                "ppl_ratio_ci_low_from_delta_nll": math.exp(delta_boot.ci_low) if math.isfinite(delta_boot.ci_low) else float("nan"),
                "ppl_ratio_ci_high_from_delta_nll": math.exp(delta_boot.ci_high) if math.isfinite(delta_boot.ci_high) else float("nan"),
            }
            for metric in ("dummy_minus_bos", "dummy_attention", "bos_attention", "output_kl"):
                metric_boot = bootstrap_mean(group[metric].to_numpy(dtype=np.float64), rng=rng, samples=samples)
                row[f"{metric}_mean"] = metric_boot.mean
                row[f"{metric}_ci_low"] = metric_boot.ci_low
                row[f"{metric}_ci_high"] = metric_boot.ci_high
            rows.append(row)
    write_csv(out / "intervention_decomposition_bootstrap.csv", rows)
    return rows


def postprocess_toy_bridge(out: Path, *, rng: np.random.Generator, samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not TOY_BRIDGE.exists():
        rows.append(
            {
                "artifact_family": "toy_bridge",
                "source_path": str(TOY_BRIDGE),
                "status": "missing_source",
            }
        )
        write_csv(out / "toy_bridge_random_control_bootstrap.csv", rows)
        return rows
    df = pd.read_csv(TOY_BRIDGE)
    for _, source in df.iterrows():
        values = np.asarray(parse_float_list(source.get("random_dummy_minus_bos_values")), dtype=np.float64)
        boot = bootstrap_mean(values, rng=rng, samples=samples)
        rows.append(
            {
                "artifact_family": "toy_bridge",
                "step": int(source["step"]),
                "source_path": str(TOY_BRIDGE),
                "resampling_unit": "random_control_run",
                "num_units": boot.n,
                "random_dummy_minus_bos_mean": boot.mean,
                "random_dummy_minus_bos_ci_low": boot.ci_low,
                "random_dummy_minus_bos_ci_high": boot.ci_high,
                "selected_dummy_minus_bos_mean": float(source["selected_dummy_minus_bos_mean"]),
                "dummy_only_dummy_minus_bos_mean": float(source["dummy_only_dummy_minus_bos_mean"]),
                "selected_minus_random_mean": float(source["selected_dummy_minus_bos_mean"]) - boot.mean,
                "note": "CI covers layer-matched random-control runs only; selected and dummy-only rows are aggregate-only in preserved source.",
            }
        )
    write_csv(out / "toy_bridge_random_control_bootstrap.csv", rows)
    return rows


def aggregate_only_status(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, path in AGGREGATE_ONLY_SOURCES.items():
        if not path.exists():
            rows.append({"artifact_family": family, "source_path": str(path), "status": "missing_source"})
            continue
        df = pd.read_csv(path)
        rows.append(
            {
                "artifact_family": family,
                "source_path": str(path),
                "status": "aggregate_only_no_bootstrap",
                "num_rows": int(len(df)),
                "columns": " ".join(str(col) for col in df.columns),
                "note": "No preserved per-example, per-window, per-subject, or multi-shard rows were found for a valid nonparametric bootstrap.",
            }
        )
    write_csv(out / "aggregate_only_status.csv", rows)
    return rows


def write_readme(
    out: Path,
    *,
    args: argparse.Namespace,
    decomp_rows: list[dict[str, Any]],
    toy_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
) -> None:
    ok_decomp = [row for row in decomp_rows if row.get("resampling_unit") == "example_idx"]
    ok_toy = [row for row in toy_rows if row.get("resampling_unit") == "random_control_run"]
    lines = [
        "# Bootstrap Postprocessing",
        "",
        "Status: complete for preserved artifacts that contain a valid CPU-only resampling unit.",
        "",
        "Command:",
        "",
        "```bash",
        f".conda-env/bin/python postprocess_bootstrap_artifacts.py --out {args.out} --seed {args.seed} --samples {args.samples}",
        "```",
        "",
        f"Code snapshot before this artifact commit: `{git_head()}`",
        "",
        "Outputs:",
        "",
        "- `intervention_decomposition_bootstrap.csv`: nonparametric example-level bootstrap CIs for Pythia-1B, Llama-3.2-3B, and Qwen3-4B intervention-decomposition per-example rows.",
        "- `toy_bridge_random_control_bootstrap.csv`: random-control-run bootstrap CIs for Figure 7 toy bridge random controls.",
        "- `aggregate_only_status.csv`: explicit status for Pile, MMLU, and repeated-prefix attack sources where only aggregate rows are preserved.",
        "- `summary.md`: compact human-readable summary.",
        "",
        "Sanity-check status:",
        "",
        f"- Intervention-decomposition bootstrap rows with example-level support: `{len(ok_decomp)}`.",
        f"- Toy-bridge random-control bootstrap rows: `{len(ok_toy)}`.",
        f"- Aggregate-only source statuses: `{len(status_rows)}`.",
        "",
        "Important limitation:",
        "",
        "Pile, MMLU, and repeated-prefix attack artifacts in the preserved main-paper bundle contain aggregate rows only. This script deliberately does not invent confidence intervals from aggregate means, single shards, or reported sample counts.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def write_summary(out: Path, decomp_rows: list[dict[str, Any]], toy_rows: list[dict[str, Any]], status_rows: list[dict[str, Any]]) -> None:
    lines = ["# Bootstrap Postprocessing Summary", ""]
    lines.append("## Intervention Decomposition")
    lines.append("")
    lines.append("| model | condition | group | n | dummy-BOS mean | 95% CI | PPL ratio | 95% CI |")
    lines.append("|---|---|---|---:|---:|---|---:|---|")
    selected = [
        row
        for row in decomp_rows
        if row.get("resampling_unit") == "example_idx"
    ]
    for row in selected:
        lines.append(
            "| {model_key} | {condition} | {candidate_group} | {num_units} | {dummy_minus_bos_mean:.6g} | [{dummy_minus_bos_ci_low:.6g}, {dummy_minus_bos_ci_high:.6g}] | {ppl_ratio_mean_from_delta_nll:.6g} | [{ppl_ratio_ci_low_from_delta_nll:.6g}, {ppl_ratio_ci_high_from_delta_nll:.6g}] |".format(**row)
        )
    lines.extend(["", "## Toy Bridge Random Controls", ""])
    lines.append("| step | n random controls | random dummy-BOS mean | 95% CI | selected dummy-BOS | selected-random |")
    lines.append("|---:|---:|---:|---|---:|---:|")
    for row in toy_rows:
        if row.get("resampling_unit") != "random_control_run":
            continue
        lines.append(
            "| {step} | {num_units} | {random_dummy_minus_bos_mean:.6g} | [{random_dummy_minus_bos_ci_low:.6g}, {random_dummy_minus_bos_ci_high:.6g}] | {selected_dummy_minus_bos_mean:.6g} | {selected_minus_random_mean:.6g} |".format(**row)
        )
    lines.extend(["", "## Aggregate-Only Sources", ""])
    lines.append("| artifact | status | rows | source |")
    lines.append("|---|---|---:|---|")
    for row in status_rows:
        lines.append(f"| {row.get('artifact_family')} | {row.get('status')} | {row.get('num_rows', '')} | `{row.get('source_path')}` |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only bootstrap postprocessing for preserved experiment artifacts.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    manifest = {
        "seed": int(args.seed),
        "samples": int(args.samples),
        "git_head": git_head(),
        "sources": {
            "intervention_decomposition": {key: str(path) for key, path in INTERVENTION_DECOMPOSITION.items()},
            "toy_bridge": str(TOY_BRIDGE),
            "aggregate_only": {key: str(path) for key, path in AGGREGATE_ONLY_SOURCES.items()},
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    decomp_rows = postprocess_intervention_decomposition(args.out, rng=rng, samples=int(args.samples))
    toy_rows = postprocess_toy_bridge(args.out, rng=rng, samples=int(args.samples))
    status_rows = aggregate_only_status(args.out)
    write_summary(args.out, decomp_rows, toy_rows, status_rows)
    write_readme(args.out, args=args, decomp_rows=decomp_rows, toy_rows=toy_rows, status_rows=status_rows)
    print(args.out / "summary.md")


if __name__ == "__main__":
    main()
