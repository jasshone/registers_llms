from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("pythia_1b", "llama3_2_3b", "qwen3_4b")
DEFAULT_DATASETS = ("wikitext_test", "pile_val")
DEFAULT_DELTAS = (0.02, 0.05, 0.10)
CONTROL_KINDS = ("early", "random")
HEAD_KINDS = ("sink_heavy", "low_sink")
TARGET_KINDS = ("sink", "early", "random")
LAYER_BANDS = ("early", "middle", "late")


class AuditResult:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def delta_matches(value: Any, expected: float, *, tol: float = 1e-9) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tol
    except Exception:
        return False


def rows_for(rows: list[dict[str, str]], **criteria: Any) -> list[dict[str, str]]:
    out = []
    for row in rows:
        matched = True
        for key, expected in criteria.items():
            value = row.get(key)
            if isinstance(expected, float):
                if not delta_matches(value, expected):
                    matched = False
                    break
            elif value != str(expected):
                matched = False
                break
        if matched:
            out.append(row)
    return out


def max_num_sequences(rows: list[dict[str, str]]) -> int:
    vals = []
    for row in rows:
        try:
            vals.append(int(float(row.get("num_sequences", "0"))))
        except ValueError:
            vals.append(0)
    return max(vals, default=0)


def audit_outputs(
    out: Path,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
    expected_heads: int = 16,
    min_sequences: int = 256,
    primary_delta: float = 0.05,
    sensitivity_min_sequences: int = 64,
    value_min_sequences: int | None = None,
    secondary_min_sequences: int | None = None,
) -> AuditResult:
    result = AuditResult()
    result.require(out.exists(), f"output directory missing: {out}")
    result.require((out / "run_summary.json").exists(), "run_summary.json missing")
    result.require((out / "causal_noop_experiment.md").exists(), "causal_noop_experiment.md missing")

    value_required = min_sequences if value_min_sequences is None else value_min_sequences
    secondary_required = min_sequences if secondary_min_sequences is None else secondary_min_sequences

    def required_sequences_for_delta(delta: float) -> int:
        return min_sequences if delta_matches(delta, primary_delta) else sensitivity_min_sequences

    if (out / "run_summary.json").exists():
        try:
            summary = json.loads((out / "run_summary.json").read_text())
        except Exception as exc:
            summary = {"rows": []}
            result.errors.append(f"run_summary.json is not valid JSON: {exc!r}")
        ok_pairs = {(row.get("model_key"), row.get("dataset")) for row in summary.get("rows", []) if row.get("status") == "ok"}
        for model in models:
            for dataset in datasets:
                result.require((model, dataset) in ok_pairs, f"run_summary lacks ok row for {model}/{dataset}")

    for model in models:
        head_path = out / model / "head_selection.json"
        result.require(head_path.exists(), f"{model}: head_selection.json missing")
        if head_path.exists():
            try:
                heads = json.loads(head_path.read_text())
            except Exception as exc:
                heads = {}
                result.errors.append(f"{model}: head_selection.json invalid JSON: {exc!r}")
            result.require(len(heads.get("sink_heavy", [])) == expected_heads, f"{model}: expected {expected_heads} sink-heavy heads")
            result.require(len(heads.get("low_sink", [])) == expected_heads, f"{model}: expected {expected_heads} low-sink heads")
            result.require(len(heads.get("all_scores", [])) >= expected_heads * 2, f"{model}: continuous head scores missing or too short")
            metadata = heads.get("metadata", {})
            result.require(isinstance(metadata, dict), f"{model}: head selection metadata missing")
            if isinstance(metadata, dict):
                result.require(metadata.get("selection_dataset") == "wikitext_train", f"{model}: head selection was not recorded as wikitext_train")
                result.require(metadata.get("dataset") == "Salesforce/wikitext", f"{model}: head selection dataset metadata is not Salesforce/wikitext")
                result.require(metadata.get("config") == "wikitext-103-raw-v1", f"{model}: head selection config metadata is not wikitext-103-raw-v1")
                result.require(metadata.get("split") == "train", f"{model}: head selection split metadata is not train")
                result.require(int(metadata.get("query_start", -1)) == 16, f"{model}: head selection query_start metadata is not 16")
                result.require(int(metadata.get("topk", -1)) == expected_heads, f"{model}: head selection topk metadata is not {expected_heads}")

        for dataset in datasets:
            prefix = f"{model}/{dataset}"
            dataset_dir = out / model / dataset
            expected_files = (
                "site_results.csv",
                "sequence_summaries.csv",
                "bootstrap_summaries.csv",
                "value_content_results.csv",
                "value_sequence_summaries.csv",
                "value_bootstrap_summaries.csv",
                "secondary_results.csv",
                "secondary_sequence_summaries.csv",
                "secondary_bootstrap_summaries.csv",
            )
            for filename in expected_files:
                result.require((dataset_dir / filename).exists(), f"{prefix}: {filename} missing")

            site_rows = read_csv_rows(dataset_dir / "site_results.csv")
            result.require(bool(site_rows), f"{prefix}: site_results.csv has no rows")
            for delta in deltas:
                for head_kind in HEAD_KINDS:
                    for target_kind in TARGET_KINDS:
                        combo = rows_for(site_rows, delta=delta, head_kind=head_kind, target_kind=target_kind)
                        result.require(bool(combo), f"{prefix}: missing primary raw rows for {head_kind}/{target_kind}/delta={delta}")
                        if combo:
                            seqs = {row.get("sequence_index") for row in combo}
                            required = required_sequences_for_delta(delta)
                            result.require(
                                len(seqs) >= required,
                                f"{prefix}: primary raw {head_kind}/{target_kind}/delta={delta} has < {required} sequences",
                            )

            primary = read_csv_rows(dataset_dir / "bootstrap_summaries.csv")
            result.require(bool(primary), f"{prefix}: bootstrap_summaries.csv has no rows")
            for delta in deltas:
                for head_kind in HEAD_KINDS:
                    for control_kind in CONTROL_KINDS:
                        combo = rows_for(primary, delta=delta, head_kind=head_kind, control_kind=control_kind, layer_band="all")
                        result.require(bool(combo), f"{prefix}: missing primary all-layer bootstrap for {head_kind}/{control_kind}/delta={delta}")
                        if combo:
                            required = required_sequences_for_delta(delta)
                            result.require(
                                max_num_sequences(combo) >= required,
                                f"{prefix}: primary {head_kind}/{control_kind}/delta={delta} has < {required} sequences",
                            )
                            for metric_field in (
                                "mean_control_minus_sink_loss",
                                "mean_control_minus_sink_logit_l2",
                                "mean_control_minus_sink_correct_prob",
                            ):
                                result.require(
                                    metric_field in combo[0],
                                    f"{prefix}: primary bootstrap missing {metric_field} for {head_kind}/{control_kind}/delta={delta}",
                                )
            present_bands = {row.get("layer_band") for row in primary if row.get("layer_band") != "all"}
            result.warn(bool(present_bands), f"{prefix}: no non-pooled layer-band primary rows")
            for band in LAYER_BANDS:
                result.warn(band in present_bands, f"{prefix}: no primary rows for {band} layer band")

            value_raw = read_csv_rows(dataset_dir / "value_content_results.csv")
            result.require(bool(value_raw), f"{prefix}: value_content_results.csv has no rows")
            for control_kind in ("random",):
                for intervention_kind in ("ordinary_value_at_sink", "sink_value_at_ordinary"):
                    combo = rows_for(value_raw, control_kind=control_kind, intervention_kind=intervention_kind)
                    result.require(bool(combo), f"{prefix}: missing value raw rows for {control_kind}/{intervention_kind}")
                    if combo:
                        seqs = {row.get("sequence_index") for row in combo}
                        result.require(
                            len(seqs) >= value_required,
                            f"{prefix}: value raw {control_kind}/{intervention_kind} has < {value_required} sequences",
                        )

            value_boot = read_csv_rows(dataset_dir / "value_bootstrap_summaries.csv")
            result.require(bool(value_boot), f"{prefix}: value_bootstrap_summaries.csv has no rows")
            for control_kind in ("random",):
                combo = rows_for(value_boot, control_kind=control_kind, layer_band="all")
                result.require(bool(combo), f"{prefix}: missing value paired all-layer bootstrap for {control_kind}")
                if combo:
                    result.require(max_num_sequences(combo) >= value_required, f"{prefix}: value paired {control_kind} has < {value_required} sequences")

            secondary = read_csv_rows(dataset_dir / "secondary_results.csv")
            result.require(bool(secondary), f"{prefix}: secondary_results.csv has no rows")
            for delta in (primary_delta,):
                for target_kind in TARGET_KINDS:
                    combo = rows_for(secondary, delta=delta, target_kind=target_kind)
                    result.require(bool(combo), f"{prefix}: missing secondary raw rows for {target_kind}/delta={delta}")
                    if combo:
                        seqs = {row.get("sequence_index") for row in combo}
                        result.require(len(seqs) >= secondary_required, f"{prefix}: secondary {target_kind}/delta={delta} has < {secondary_required} sequences")

            secondary_boot = read_csv_rows(dataset_dir / "secondary_bootstrap_summaries.csv")
            result.require(bool(secondary_boot), f"{prefix}: secondary_bootstrap_summaries.csv has no rows")
            for delta in (primary_delta,):
                for control_kind in CONTROL_KINDS:
                    combo = rows_for(secondary_boot, delta=delta, control_kind=control_kind)
                    result.require(bool(combo), f"{prefix}: missing secondary paired bootstrap for {control_kind}/delta={delta}")
                    if combo:
                        result.require(max_num_sequences(combo) >= secondary_required, f"{prefix}: secondary paired {control_kind}/delta={delta} has < {secondary_required} sequences")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit completed causal no-op experiment outputs.")
    parser.add_argument("--out", type=Path, default=Path("outputs/causal_noop_experiment"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--deltas", nargs="+", type=float, default=list(DEFAULT_DELTAS))
    parser.add_argument("--expected-heads", type=int, default=16)
    parser.add_argument("--min-sequences", type=int, default=256)
    parser.add_argument("--primary-delta", type=float, default=0.05)
    parser.add_argument("--sensitivity-min-sequences", type=int, default=64)
    parser.add_argument("--value-min-sequences", type=int, default=128)
    parser.add_argument("--secondary-min-sequences", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_outputs(
        args.out,
        models=tuple(args.models),
        datasets=tuple(args.datasets),
        deltas=tuple(args.deltas),
        expected_heads=int(args.expected_heads),
        min_sequences=int(args.min_sequences),
        primary_delta=float(args.primary_delta),
        sensitivity_min_sequences=int(args.sensitivity_min_sequences),
        value_min_sequences=int(args.value_min_sequences),
        secondary_min_sequences=int(args.secondary_min_sequences),
    )
    for warning in result.warnings:
        print(f"[warn] {warning}")
    if result.errors:
        for error in result.errors:
            print(f"[error] {error}")
        return 1
    print(f"causal noop output audit passed: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
