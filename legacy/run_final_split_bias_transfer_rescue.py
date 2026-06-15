from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.bias_transfer import (
    CONFIGS,
    DEFAULT_OUT,
    PPL_LIMIT,
    RESCUE_TOPKS,
    SCALE_PRESETS,
    WINDOWS_PER_SPLIT,
    audit_failure_reason,
    read_csv,
    run_rescue,
    scale_preset,
)
from sink_neurons.env import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescue failed final split bias-transfer rows with the shared public pipeline implementation.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--windows-per-split", type=int, default=WINDOWS_PER_SPLIT)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--masks", nargs="*", default=None)
    parser.add_argument("--topks", nargs="*", type=int, default=None)
    parser.add_argument("--scales", nargs="*", type=float, default=None)
    parser.add_argument("--scale-preset", choices=sorted(SCALE_PRESETS), default="rescue")
    parser.add_argument("--source-abs-bonus", type=float, default=0.1)
    parser.add_argument("--sink-source", choices=("attention_sink", "massive_activation"), default="attention_sink")
    parser.add_argument("--massive-abs-threshold", type=float, default=100.0)
    parser.add_argument("--massive-median-ratio", type=float, default=1000.0)
    parser.add_argument("--massive-min-future-queries", type=int, default=32)
    parser.add_argument("--massive-exclude-positions", type=int, nargs="*", default=[])
    args = parser.parse_args()
    args.force = True
    args.smoke = False
    args.bias_windows = args.windows_per_split
    args.rescue = "always"
    args.rescue_scale_preset = args.scale_preset
    args.scale_preset = "core"
    args.active_topks = []
    args.active_rescue_topks = sorted(set(args.topks or RESCUE_TOPKS))
    args.active_scales = []
    args.active_rescue_scales = sorted(set(args.scales if args.scales else scale_preset(args.rescue_scale_preset)))
    if any(topk <= 0 for topk in args.active_rescue_topks):
        raise SystemExit("all --topks values must be positive")
    return args


def model_cfg(key: str):
    for cfg in CONFIGS:
        if cfg.key == key:
            return cfg
    raise KeyError(key)


def failure_reason(root: Path, key: str) -> str:
    rows = read_csv(root / key / "test_result.csv")
    reason = audit_failure_reason(dict(rows[0]) if rows else None)
    return reason or f"manual rescue requested; normal result already passes ppl<={PPL_LIMIT:g} audit"


def main() -> None:
    configure_runtime()
    args = parse_args()
    for key in args.models:
        run_rescue(model_cfg(key), args.out, args, failure_reason(args.out, key))


if __name__ == "__main__":
    main()
