from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID, load_model
from sink_neurons.plotting import plot_bos_score_histogram
from sink_neurons.scoring import save_bos_scores_artifact, score_bos_neurons
from sink_neurons.windows import load_windows_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score BOS-associated MLP neurons.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/scores/bos_scores.pt"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    windows, _payload = load_windows_artifact(args.windows)
    model = load_model(args.model_id, eager_attention=True)
    result = score_bos_neurons(
        model,
        windows,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    save_bos_scores_artifact(
        args.output,
        result=result,
        model_id=args.model_id,
        window_artifact=args.windows,
        batch_size=args.batch_size,
    )
    plot_bos_score_histogram(result.scores, args.plots_dir / "bos_score_histogram.png")
    print(f"Saved BOS scores to {args.output}")


if __name__ == "__main__":
    main()
