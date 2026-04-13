from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.artifacts import load_pt
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID, load_model
from sink_neurons.plotting import plot_attention_relocation, plot_norm_relocation
from sink_neurons.relocation import run_relocation_experiment, save_relocation_artifact
from sink_neurons.windows import load_windows_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BOS relocation with one cached selection.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/relocation/relocation.pt"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    windows, _payload = load_windows_artifact(args.windows)
    selection_payload = load_pt(args.selection)
    selected = selection_payload["tensors"]["selected"]

    model = load_model(args.model_id, eager_attention=True)
    result = run_relocation_experiment(
        model,
        windows,
        selected,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    save_relocation_artifact(
        args.output,
        result=result,
        model_id=args.model_id,
        window_artifact=args.windows,
        selection_artifact=args.selection,
        batch_size=args.batch_size,
    )
    plot_norm_relocation(
        result.bos_norm_before,
        result.bos_norm_after,
        result.dummy_norm_after,
        args.plots_dir / "norm_relocation.png",
    )
    plot_attention_relocation(
        result.bos_attention_before,
        result.bos_attention_after,
        result.dummy_attention_after,
        args.plots_dir / "attention_relocation.png",
    )
    print(f"Saved relocation metrics to {args.output}")


if __name__ == "__main__":
    main()
