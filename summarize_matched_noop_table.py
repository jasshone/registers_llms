from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_INPUTS = {
    "pythia_1b": Path("outputs/causal_noop_full_wikitext_20260714/pythia_1b/wikitext_test/site_results.csv"),
    "llama3_2_3b": Path("outputs/causal_noop_full_wikitext_20260714/llama3_2_3b/wikitext_test/site_results.csv"),
    "mistral_7b_v0_1": Path("outputs/causal_noop_mistral_wikitext_fixed_20260714/mistral_7b_v0_1/wikitext_test/site_results.csv"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_ci(values: list[float], *, seed: int, samples: int) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_model(
    *,
    model: str,
    path: Path,
    delta: float,
    head_kind: str,
    ordinary_target: str,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = read_rows(path)
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    skipped_rows = 0
    for row in rows:
        if row["head_kind"] != head_kind:
            continue
        if abs(float(row["delta"]) - delta) > 1e-12:
            continue
        key = (row["sequence_index"], row["query_position"], row["layer"], row["head"])
        grouped[key][row["target_kind"]] = row

    per_sequence: dict[str, list[dict[str, float]]] = defaultdict(list)
    raw_pairs: list[dict[str, object]] = []
    total_pairs = 0
    for (seq, query, layer, head), targets in sorted(grouped.items(), key=lambda item: tuple(int(x) for x in item[0])):
        sink = targets.get("sink")
        ordinary = targets.get(ordinary_target)
        if sink is None or ordinary is None:
            continue
        total_pairs += 1
        if sink["skipped"] == "True" or ordinary["skipped"] == "True":
            skipped_rows += 1
            continue
        sink_nll = float(sink["delta_loss"])
        ordinary_nll = float(ordinary["delta_loss"])
        sink_kl = float(sink["kl_clean_intervened"])
        ordinary_kl = float(ordinary["kl_clean_intervened"])
        sink_l2 = float(sink["logit_l2"])
        ordinary_l2 = float(ordinary["logit_l2"])
        item = {
            "sink_delta_nll": sink_nll,
            "ordinary_delta_nll": ordinary_nll,
            "paired_delta_nll": ordinary_nll - sink_nll,
            "sink_kl": sink_kl,
            "ordinary_kl": ordinary_kl,
            "paired_kl": ordinary_kl - sink_kl,
            "sink_logit_l2": sink_l2,
            "ordinary_logit_l2": ordinary_l2,
            "paired_logit_l2": ordinary_l2 - sink_l2,
        }
        per_sequence[seq].append(item)
        raw_pairs.append(
            {
                "model": model,
                "sequence_index": int(seq),
                "query_position": int(query),
                "layer": int(layer),
                "head": int(head),
                "head_kind": head_kind,
                "ordinary_target": ordinary_target,
                "delta": delta,
                **item,
            }
        )

    sequence_rows = []
    for seq, items in sorted(per_sequence.items(), key=lambda item: int(item[0])):
        seq_row = {"model": model, "sequence_index": int(seq), "num_pairs": len(items)}
        for key in (
            "sink_delta_nll",
            "ordinary_delta_nll",
            "paired_delta_nll",
            "sink_kl",
            "ordinary_kl",
            "paired_kl",
            "sink_logit_l2",
            "ordinary_logit_l2",
            "paired_logit_l2",
        ):
            seq_row[key] = mean(item[key] for item in items)
        sequence_rows.append(seq_row)

    paired_nll = [float(row["paired_delta_nll"]) for row in sequence_rows]
    paired_kl = [float(row["paired_kl"]) for row in sequence_rows]
    paired_l2 = [float(row["paired_logit_l2"]) for row in sequence_rows]
    ci_nll = bootstrap_ci(paired_nll, seed=seed, samples=bootstrap_samples)
    ci_kl = bootstrap_ci(paired_kl, seed=seed + 1, samples=bootstrap_samples)
    ci_l2 = bootstrap_ci(paired_l2, seed=seed + 2, samples=bootstrap_samples)

    summary = {
        "model": model,
        "source_csv": str(path),
        "delta": delta,
        "head_kind": head_kind,
        "ordinary_target": ordinary_target,
        "sequence_level_units": len(sequence_rows),
        "site_pairs_used": len(raw_pairs),
        "site_pairs_available": total_pairs,
        "site_pairs_skipped": skipped_rows,
        "sink_delta_nll": mean(float(row["sink_delta_nll"]) for row in sequence_rows),
        "ordinary_delta_nll": mean(float(row["ordinary_delta_nll"]) for row in sequence_rows),
        "paired_delta_nll": mean(paired_nll),
        "paired_delta_nll_ci_low": ci_nll[0],
        "paired_delta_nll_ci_high": ci_nll[1],
        "fraction_positive_delta_nll": sum(value > 0.0 for value in paired_nll) / len(paired_nll),
        "sink_kl": mean(float(row["sink_kl"]) for row in sequence_rows),
        "ordinary_kl": mean(float(row["ordinary_kl"]) for row in sequence_rows),
        "paired_kl": mean(paired_kl),
        "paired_kl_ci_low": ci_kl[0],
        "paired_kl_ci_high": ci_kl[1],
        "fraction_positive_kl": sum(value > 0.0 for value in paired_kl) / len(paired_kl),
        "sink_logit_l2": mean(float(row["sink_logit_l2"]) for row in sequence_rows),
        "ordinary_logit_l2": mean(float(row["ordinary_logit_l2"]) for row in sequence_rows),
        "paired_logit_l2": mean(paired_l2),
        "paired_logit_l2_ci_low": ci_l2[0],
        "paired_logit_l2_ci_high": ci_l2[1],
        "fraction_positive_logit_l2": sum(value > 0.0 for value in paired_l2) / len(paired_l2),
    }
    return summary, sequence_rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Matched No-Op Sink vs Ordinary Table",
        "",
        "Primary comparison: same sequence, query, layer, head, and injected attention mass; target is either the sink token or a matched ordinary token. Rows are averaged at the sequence level before bootstrapping.",
        "",
        f"- Delta: {args.delta}",
        f"- Head set: {args.head_kind}",
        f"- Ordinary target: {args.ordinary_target}",
        f"- Bootstrap samples: {args.bootstrap_samples}",
        "",
        "Positive paired differences mean ordinary-token redirection was more disruptive than sink-token redirection.",
        "",
        "| Model | Sink ΔNLL | Ordinary ΔNLL | Ordinary - sink ΔNLL | 95% CI | Fraction positive | Sink KL | Ordinary KL | Ordinary - sink KL | 95% CI | n seq |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {sink_delta_nll:.6g} | {ordinary_delta_nll:.6g} | {paired_delta_nll:.6g} | [{paired_delta_nll_ci_low:.6g}, {paired_delta_nll_ci_high:.6g}] | {fraction_positive_delta_nll:.3f} | {sink_kl:.6g} | {ordinary_kl:.6g} | {paired_kl:.6g} | [{paired_kl_ci_low:.6g}, {paired_kl_ci_high:.6g}] | {sequence_level_units} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Logit-change norm check:",
            "",
            "| Model | Sink logit L2 | Ordinary logit L2 | Ordinary - sink logit L2 | 95% CI | Fraction positive |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {model} | {sink_logit_l2:.6g} | {ordinary_logit_l2:.6g} | {paired_logit_l2:.6g} | [{paired_logit_l2_ci_low:.6g}, {paired_logit_l2_ci_high:.6g}] | {fraction_positive_logit_l2:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The ΔNLL confidence intervals cross zero for the representative models, so this is not strong evidence that sink redirection is reliably less disruptive on next-token loss.",
            "- KL and logit-change norm are sometimes positive, but the evidence is not uniformly strong enough to support a broad No-Op mechanism claim from this table alone.",
            "- The narrower value-content claim remains safer unless a larger or more targeted matched run produces sequence-level CIs above zero.",
            "",
            "Skipped site pairs were excluded only when either the sink or ordinary row could not receive the requested fixed attention mass because the target already had too much attention.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched No-Op sink-vs-ordinary results.")
    parser.add_argument("--out", type=Path, default=Path("outputs/matched_noop_clean_table_20260715"))
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--head-kind", default="sink_heavy", choices=["sink_heavy", "low_sink"])
    parser.add_argument("--ordinary-target", default="random", choices=["early", "random"])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    summaries = []
    sequence_rows = []
    for idx, (model, path) in enumerate(DEFAULT_INPUTS.items()):
        if not path.exists():
            raise FileNotFoundError(path)
        summary, seq_rows = summarize_model(
            model=model,
            path=path,
            delta=args.delta,
            head_kind=args.head_kind,
            ordinary_target=args.ordinary_target,
            seed=args.seed + idx * 100,
            bootstrap_samples=args.bootstrap_samples,
        )
        summaries.append(summary)
        sequence_rows.extend(seq_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "matched_noop_model_summary.csv", summaries)
    write_csv(args.out / "matched_noop_sequence_effects.csv", sequence_rows)
    write_report(args.out / "matched_noop_clean_table.md", summaries, args)
    (args.out / "matched_noop_config.json").write_text(
        json.dumps(
            {
                "delta": args.delta,
                "head_kind": args.head_kind,
                "ordinary_target": args.ordinary_target,
                "seed": args.seed,
                "bootstrap_samples": args.bootstrap_samples,
                "inputs": {model: str(path) for model, path in DEFAULT_INPUTS.items()},
                "unit_of_analysis": "sequence-level average over matched site pairs",
                "paired_difference": "ordinary target disruption minus sink target disruption",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
