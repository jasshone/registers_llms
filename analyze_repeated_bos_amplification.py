from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("outputs/repeated_bos_amplification_stats_20260715")
DEFAULT_SOURCES = {
    "pythia_1b": Path("outputs/preserved_main_paper_figures/sources/figure8_repeated_bos_prefix_metrics.csv"),
    "llama3_2_3b": Path("outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714/repeated_bos_prefix_metrics.csv"),
    "qwen3_4b": Path("outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714/repeated_bos_prefix_metrics.csv"),
    "mistral_7b_v0_1": Path("outputs/sink_hijacking_attack/repeated_bos_prefix/mistral_7b_v0_1_20260714/repeated_bos_prefix_metrics.csv"),
}
METRICS = ("ppl_ratio_vs_clean", "inserted_attention_mean", "inserted_minus_bos_mean")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def mean(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if len(vals) < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_summary(diffs: list[float], *, rng: random.Random, samples: int) -> dict[str, Any]:
    vals = [x for x in diffs if math.isfinite(x)]
    n = len(vals)
    if n == 0:
        return {"n_pairs": 0}
    m = mean(vals)
    sd = stdev(vals)
    se = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else float("nan")
    z = m / se if se and math.isfinite(se) and se > 0 else float("nan")
    p_norm = 2.0 * (1.0 - normal_cdf(abs(z))) if math.isfinite(z) else float("nan")
    positive = sum(x > 0 for x in vals)
    nonzero = sum(x != 0 for x in vals)
    k = min(positive, nonzero - positive)
    sign_p = float("nan")
    if nonzero:
        # Exact two-sided sign test under p=0.5 for small n.
        sign_p = min(1.0, 2.0 * sum(math.comb(nonzero, i) for i in range(k + 1)) / (2 ** nonzero))
    boots = []
    for _ in range(samples):
        boots.append(mean([vals[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    lo = boots[max(0, int(0.025 * samples) - 1)]
    hi = boots[min(samples - 1, int(0.975 * samples))]
    return {
        "n_pairs": n,
        "mean_diff": m,
        "std_diff": sd,
        "ci_low": lo,
        "ci_high": hi,
        "fraction_positive": positive / n,
        "sign_test_p_two_sided": sign_p,
        "normal_approx_p_two_sided": p_norm,
    }


def row_by_condition(rows: list[dict[str, str]], condition: str, count: int) -> dict[str, str] | None:
    matches = [r for r in rows if r.get("condition") == condition and int(float(r.get("count", "-1"))) == count]
    if not matches:
        return None
    if condition == "random_copy_bos":
        # Average random controls if more than one is present at the count.
        out = dict(matches[0])
        for metric in METRICS:
            out[metric] = str(mean([fnum(r, metric) for r in matches]))
        return out
    return matches[0]


def analyze_source(model: str, path: Path, *, rng: random.Random, samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(path)
    counts = sorted({int(float(r["count"])) for r in rows if r.get("condition") != "clean"})
    pair_rows: list[dict[str, Any]] = []
    for count in counts:
        selected = row_by_condition(rows, "selected_copy_bos", count)
        bos = row_by_condition(rows, "bos_repeat_only", count)
        random_row = row_by_condition(rows, "random_copy_bos", count)
        ordinary = row_by_condition(rows, "ordinary_repeat_only", count)
        if selected is None or bos is None:
            continue
        controls = {"bos_repeat_only": bos, "random_copy_bos": random_row, "ordinary_repeat_only": ordinary}
        for control, crow in controls.items():
            if crow is None:
                continue
            for metric in METRICS:
                s = fnum(selected, metric)
                c = fnum(crow, metric)
                pair_rows.append({
                    "model": model,
                    "source_path": str(path),
                    "count": count,
                    "control": control,
                    "metric": metric,
                    "selected_value": s,
                    "control_value": c,
                    "selected_minus_control": s - c,
                })
    summary_rows: list[dict[str, Any]] = []
    for control in sorted({r["control"] for r in pair_rows}):
        for metric in METRICS:
            diffs = [float(r["selected_minus_control"]) for r in pair_rows if r["control"] == control and r["metric"] == metric]
            summary_rows.append({"model": model, "control": control, "metric": metric, **paired_summary(diffs, rng=rng, samples=samples)})
    return pair_rows, summary_rows


def write_md(path: Path, summary_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], sources: dict[str, Path]) -> None:
    lines = [
        "# Repeated-BOS Amplification Statistics",
        "",
        "This postprocesses existing repeated-BOS aggregate outputs. The paired unit is prefix count within each model, not individual windows, because the repeated-BOS runner preserved only aggregate metrics. Positive differences mean `selected_copy_bos` is larger than the named control at the same prefix count.",
        "",
        "Sources:",
    ]
    for model, path0 in sources.items():
        lines.append(f"- `{model}`: `{path0}`")
    lines.extend([
        "",
        "## Summary",
        "",
        "| model | control | metric | n pairs | mean selected-control | 95% bootstrap CI | fraction positive | sign p | normal approx p |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ])
    for r in summary_rows:
        lines.append(
            "| {model} | {control} | {metric} | {n_pairs} | {mean_diff:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | {fraction_positive:.3g} | {sign_test_p_two_sided:.6g} | {normal_approx_p_two_sided:.6g} |".format(**r)
        )
    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- This is sufficient for a count-sweep amplification check on the preserved artifacts.",
        "- It is not a per-window uncertainty estimate; that would require rerunning the attack runner with per-window outputs or modifying the runner to emit them.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess repeated-BOS aggregate outputs for selected-feature amplification.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    rng = random.Random(int(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {model: path for model, path in DEFAULT_SOURCES.items() if path.exists()}
    all_pairs: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for model, path in sources.items():
        pairs, summaries = analyze_source(model, path, rng=rng, samples=int(args.samples))
        all_pairs.extend(pairs)
        all_summaries.extend(summaries)
    # Cross-model/count paired summary pools each model-count as one unit.
    for control in sorted({r["control"] for r in all_pairs}):
        for metric in METRICS:
            diffs = [float(r["selected_minus_control"]) for r in all_pairs if r["control"] == control and r["metric"] == metric]
            all_summaries.append({"model": "pooled_model_count", "control": control, "metric": metric, **paired_summary(diffs, rng=rng, samples=int(args.samples))})
    write_csv(args.out / "repeated_bos_amplification_pairs.csv", all_pairs)
    write_csv(args.out / "repeated_bos_amplification_summary.csv", all_summaries)
    write_md(args.out / "repeated_bos_amplification_summary.md", all_summaries, all_pairs, sources)
    print(args.out / "repeated_bos_amplification_summary.md")


if __name__ == "__main__":
    main()
