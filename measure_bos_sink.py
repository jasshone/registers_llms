from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.diagnostics import measure_bos_diagnostics, save_bos_diagnostics_artifact
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID, load_model
from sink_neurons.plotting import plot_mean_bos_attention_by_layer, plot_mean_token_norm_heatmap
from sink_neurons.windows import load_windows_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure BOS sink diagnostics on cached windows.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/bos_diagnostics.pt"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    windows, _payload = load_windows_artifact(args.windows)
    model = load_model(args.model_id, eager_attention=True)
    result = measure_bos_diagnostics(
        model,
        windows,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    save_bos_diagnostics_artifact(
        args.output,
        result=result,
        model_id=args.model_id,
        window_artifact=args.windows,
        batch_size=args.batch_size,
    )
    plot_mean_token_norm_heatmap(result.mean_token_norms, args.plots_dir / "mean_token_norm_heatmap.png")
    plot_mean_bos_attention_by_layer(
        result.mean_bos_attention_by_layer,
        args.plots_dir / "mean_bos_attention_by_layer.png",
    )
    print(f"Saved diagnostics to {args.output}")


if __name__ == "__main__":
    main()
