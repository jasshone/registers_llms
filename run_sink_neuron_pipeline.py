from __future__ import annotations

import argparse
import json
from pathlib import Path

from sink_neurons.bias_transfer import (
    CONFIGS,
    DEFAULT_OUT,
    FAMILIES,
    MODEL_SETS,
    NORMAL_TOPKS,
    REPRODUCTION_PRESETS,
    RESCUE_TOPKS,
    SCALE_PRESETS,
    WINDOWS_PER_SPLIT,
    run_pipeline,
    scale_preset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public final-split sink-neuron bias-transfer pipeline with standardized scale presets."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-set", choices=sorted(MODEL_SETS), help="Curated named model set, e.g. gpt2-family or paper-core.")
    parser.add_argument("--model-group", choices=sorted(MODEL_SETS), help="Deprecated alias for --model-set.")
    parser.add_argument("--list-model-sets", action="store_true", help="Print curated model sets and exit.")
    parser.add_argument("--list-model-groups", action="store_true", help="Deprecated alias for --list-model-sets.")
    parser.add_argument("--run-profile", choices=("smoke", "standard"), default=None, help="Run size profile. smoke uses 4 train/val/test windows; standard uses 64.")
    parser.add_argument("--preset", choices=sorted(REPRODUCTION_PRESETS), help="Legacy compatibility alias that sets model group/profile/scale/rescue together.")
    parser.add_argument("--list-presets", action="store_true", help="Print legacy reproduction preset aliases and exit.")
    parser.add_argument("--models", nargs="*", help="Optional explicit model keys to run. Overrides --model-set and --preset models when supplied.")
    parser.add_argument("--families", nargs="*", choices=FAMILIES, help="Sweep every configured model in one or more families, e.g. gpt2 or pythia.")
    parser.add_argument("--list-families", action="store_true", help="Print configured families and their model keys.")
    parser.add_argument("--force", action="store_true", help="Rerun models even if test_result.csv exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run configuration and exit without loading models or datasets.")
    parser.add_argument("--smoke", action="store_true", help="Deprecated alias for --run-profile smoke.")
    parser.add_argument("--windows-per-split", type=int, default=None)
    parser.add_argument("--bias-windows", type=int, default=None)
    parser.add_argument("--topks", nargs="*", type=int, default=None, help="Validation top-k grid. Defaults to the standard final-split grid.")
    parser.add_argument("--rescue-topks", nargs="*", type=int, default=None, help="Rescue top-k grid. Defaults to the standard rescue grid.")
    parser.add_argument("--scales", nargs="*", type=float, default=None, help="Explicit validation scale grid. Overrides --scale-preset.")
    parser.add_argument("--rescue-scales", nargs="*", type=float, default=None, help="Explicit rescue scale grid. Overrides --rescue-scale-preset.")
    parser.add_argument("--scale-preset", choices=sorted(SCALE_PRESETS), default=None)
    parser.add_argument("--rescue-scale-preset", choices=sorted(SCALE_PRESETS), default=None)
    parser.add_argument("--list-scale-presets", action="store_true", help="Print standardized scale presets and exit.")
    parser.add_argument("--rescue", choices=("failed", "always", "off"), default=None)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional rescue layer-group filter.")
    parser.add_argument("--masks", nargs="*", default=None, help="Optional rescue mask filter.")
    parser.add_argument(
        "--sink-source",
        choices=("attention_sink", "massive_activation"),
        default="attention_sink",
        help="How to build non-BOS source-token masks for pipeline_sinks_plus_bos.",
    )
    parser.add_argument("--massive-abs-threshold", type=float, default=100.0)
    parser.add_argument("--massive-median-ratio", type=float, default=1000.0)
    parser.add_argument("--massive-min-future-queries", type=int, default=32)
    parser.add_argument("--massive-exclude-positions", type=int, nargs="*", default=[])
    parser.add_argument("--source-abs-bonus", type=float, default=0.1)
    args = parser.parse_args()

    if args.list_scale_presets:
        for name in sorted(SCALE_PRESETS):
            values = " ".join(f"{value:g}" for value in SCALE_PRESETS[name])
            print(f"{name}: {values}")
        raise SystemExit(0)
    if args.list_families:
        for family in FAMILIES:
            models = [cfg.key for cfg in CONFIGS if cfg.family == family]
            print(f"{family}: models=[{' '.join(models)}]")
        raise SystemExit(0)
    if args.model_group and not args.model_set:
        args.model_set = args.model_group
    if args.list_model_sets or args.list_model_groups:
        for name in sorted(MODEL_SETS):
            print(f"{name}: models=[{' '.join(MODEL_SETS[name])}]")
        raise SystemExit(0)
    if args.list_presets:
        for name in sorted(REPRODUCTION_PRESETS):
            preset = REPRODUCTION_PRESETS[name]
            models = " ".join(preset.models)
            print(
                f"{name}: {preset.description} models=[{models}] "
                f"windows_per_split={preset.windows_per_split} bias_windows={preset.bias_windows} "
                f"scale_preset={preset.scale_preset} rescue={preset.rescue}"
            )
        raise SystemExit(0)

    preset = REPRODUCTION_PRESETS.get(args.preset) if args.preset else None
    if args.smoke and args.run_profile is None:
        args.run_profile = "smoke"
    if preset is not None:
        if args.run_profile is None:
            args.run_profile = "smoke" if preset.windows_per_split <= 4 else "standard"
        if args.windows_per_split is None:
            args.windows_per_split = preset.windows_per_split
        if args.bias_windows is None:
            args.bias_windows = preset.bias_windows
        if args.scale_preset is None:
            args.scale_preset = preset.scale_preset
        if args.rescue_scale_preset is None:
            args.rescue_scale_preset = preset.rescue_scale_preset
        if args.rescue is None:
            args.rescue = preset.rescue
        if args.topks is None:
            args.topks = list(preset.topks)
        if args.rescue_topks is None:
            args.rescue_topks = list(preset.rescue_topks)

    if args.run_profile is None:
        args.run_profile = "standard"
    if args.windows_per_split is None:
        args.windows_per_split = 4 if args.run_profile == "smoke" else WINDOWS_PER_SPLIT
    if args.bias_windows is None:
        args.bias_windows = args.windows_per_split
    if args.scale_preset is None:
        args.scale_preset = "core"
    if args.rescue_scale_preset is None:
        args.rescue_scale_preset = "rescue"
    if args.rescue is None:
        args.rescue = "off" if args.run_profile == "smoke" else "failed"

    if args.models:
        pass
    elif args.model_set:
        args.models = list(MODEL_SETS[args.model_set])
    elif preset is not None:
        args.models = list(preset.models)
        if args.model_set is None:
            for set_name, set_models in MODEL_SETS.items():
                if tuple(args.models) == tuple(set_models):
                    args.model_set = set_name
                    break
    elif not args.families:
        args.model_set = "gpt2"
        args.models = list(MODEL_SETS[args.model_set])

    if args.models:
        known = {cfg.key for cfg in CONFIGS}
        unknown = sorted(set(args.models) - known)
        if unknown:
            raise SystemExit(f"unknown model key(s): {', '.join(unknown)}")
    if args.windows_per_split < 1:
        raise SystemExit("--windows-per-split must be positive")
    if args.bias_windows < 1:
        raise SystemExit("--bias-windows must be positive")

    args.active_topks = sorted(set(args.topks or NORMAL_TOPKS))
    args.active_rescue_topks = sorted(set(args.rescue_topks or RESCUE_TOPKS))
    args.active_scales = sorted(set(args.scales if args.scales else scale_preset(args.scale_preset)))
    args.active_rescue_scales = sorted(set(args.rescue_scales if args.rescue_scales else scale_preset(args.rescue_scale_preset)))

    if any(topk <= 0 for topk in args.active_topks + args.active_rescue_topks):
        raise SystemExit("all top-k values must be positive")
    if args.massive_min_future_queries < 1:
        raise SystemExit("--massive-min-future-queries must be positive")
    return args


def resolved_config(args: argparse.Namespace) -> dict[str, object]:
    selected_models = args.models or [cfg.key for cfg in CONFIGS if not args.families or cfg.family in set(args.families)]
    return {
        "model_set": args.model_set,
        "run_profile": args.run_profile,
        "preset": args.preset,
        "out": str(args.out),
        "models": selected_models,
        "families": args.families or [],
        "windows_per_split": args.windows_per_split,
        "bias_windows": args.bias_windows,
        "topks": args.active_topks,
        "scale_preset": args.scale_preset,
        "scales": args.active_scales,
        "rescue": args.rescue,
        "rescue_topks": args.active_rescue_topks,
        "rescue_scale_preset": args.rescue_scale_preset,
        "rescue_scales": args.active_rescue_scales,
        "sink_source": args.sink_source,
        "source_abs_bonus": args.source_abs_bonus,
    }


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(resolved_config(args), indent=2, sort_keys=True))
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
