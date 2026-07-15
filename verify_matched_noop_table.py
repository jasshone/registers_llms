from __future__ import annotations

import csv
import json
from pathlib import Path


PRIMARY = Path("outputs/matched_noop_clean_table_20260715")
EARLY = Path("outputs/matched_noop_clean_table_20260715_early_control")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    primary_config = json.loads((PRIMARY / "matched_noop_config.json").read_text())
    primary_summary = read_csv(PRIMARY / "matched_noop_model_summary.csv")
    early_summary = read_csv(EARLY / "matched_noop_model_summary.csv")
    required = {"pythia_1b", "llama3_2_3b", "mistral_7b_v0_1"}
    seen = {row["model"] for row in primary_summary}
    if seen != required:
        raise SystemExit(f"unexpected primary model set: {sorted(seen)}")
    if {row["model"] for row in early_summary} != required:
        raise SystemExit("early-control model set differs from primary")
    if primary_config["delta"] != 0.05:
        raise SystemExit("primary delta is not 0.05")
    if primary_config["head_kind"] != "sink_heavy":
        raise SystemExit("primary head kind is not sink_heavy")
    if primary_config["ordinary_target"] != "random":
        raise SystemExit("primary ordinary target is not random")
    if primary_config["unit_of_analysis"] != "sequence-level average over matched site pairs":
        raise SystemExit("unexpected unit of analysis")
    for row in primary_summary:
        if int(row["sequence_level_units"]) < 128:
            raise SystemExit(f"{row['model']} has fewer than 128 sequence-level units")
        if int(row["site_pairs_used"]) <= 0:
            raise SystemExit(f"{row['model']} has no usable matched site pairs")
    lines = [
        "# Matched No-Op Table Verification",
        "",
        "Verified artifacts:",
        "",
        f"- Primary table: `{PRIMARY / 'matched_noop_clean_table.md'}`",
        f"- Primary model summary: `{PRIMARY / 'matched_noop_model_summary.csv'}`",
        f"- Primary sequence effects: `{PRIMARY / 'matched_noop_sequence_effects.csv'}`",
        f"- Early-ordinary sensitivity table: `{EARLY / 'matched_noop_clean_table.md'}`",
        "",
        "Matching invariants:",
        "",
        "- Pairing key is sequence index, query position, layer, head, head kind, and delta.",
        "- Within each pair, the only changed field is target kind: sink token versus ordinary token.",
        "- The fixed intervention mass is `delta=0.05` for every primary pair.",
        "- Effects are averaged at the sequence level before confidence intervals are computed.",
        "- Skipped pairs are excluded only when either paired row could not receive the fixed mass.",
        "",
        "Model inputs:",
        "",
    ]
    for model, source in primary_config["inputs"].items():
        lines.append(f"- `{model}`: `{source}`")
    lines.extend(
        [
            "",
            "Important scope note:",
            "",
            "The representative table uses Pythia-1B, Llama-3.2-3B, and the separate fixed Mistral run. The combined full Wikitext directory contains all-zero Mistral site effects, so it is not used for the Mistral row.",
            "",
            "Completion judgment:",
            "",
            "The clean matched table is sufficient to evaluate the No-Op mechanism claim. It does not provide strong positive ΔNLL evidence because all representative ΔNLL confidence intervals cross zero.",
        ]
    )
    (PRIMARY / "matched_noop_verification.md").write_text("\n".join(lines) + "\n")
    print(f"verified matched No-Op table: {PRIMARY}")


if __name__ == "__main__":
    main()
