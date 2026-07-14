from __future__ import annotations

import argparse
from pathlib import Path

import torch

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
    parser.add_argument(
        "--dummy-init",
        choices=("zero", "bos"),
        default="zero",
        help="Initialization for the inserted dummy slot.",
    )
    parser.add_argument(
        "--relocation-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to selected activations when they are written into the dummy slot.",
    )
    parser.add_argument(
        "--relocation-fraction",
        type=float,
        default=1.0,
        help="Fraction of selected activations moved into dummy slots; the remainder stays at original positions.",
    )
    parser.add_argument(
        "--num-dummy-tokens",
        type=int,
        default=1,
        help="Number of dummy slots to insert after BOS.",
    )
    parser.add_argument(
        "--assignment-strategy",
        choices=("round_robin", "layer_blocked", "magnitude_balanced"),
        default="round_robin",
        help="How selected neurons are assigned to dummy slots.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.num_dummy_tokens < 1:
        raise SystemExit("--num-dummy-tokens must be at least 1")
    if not 0.0 <= args.relocation_fraction <= 1.0:
        raise SystemExit("--relocation-fraction must be in [0, 1]")
    return args


def _load_selected_scores(selection_payload: dict) -> torch.Tensor | None:
    score_artifact = selection_payload.get("metadata", {}).get("score_artifact")
    if not score_artifact:
        return None
    score_payload = load_pt(Path(score_artifact))
    scores = score_payload.get("tensors", {}).get("scores")
    selected = selection_payload["tensors"]["selected"]
    if scores is None:
        return None
    return scores[selected[:, 0], selected[:, 1]]


def main() -> None:
    configure_runtime()
    args = parse_args()
    windows, _payload = load_windows_artifact(args.windows)
    selection_payload = load_pt(args.selection)
    selected = selection_payload["tensors"]["selected"]
    selected_scores = _load_selected_scores(selection_payload) if args.assignment_strategy == "magnitude_balanced" else None

    model = load_model(args.model_id, eager_attention=True)
    result = run_relocation_experiment(
        model,
        windows,
        selected,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        dummy_init=args.dummy_init,
        num_dummy_tokens=args.num_dummy_tokens,
        relocation_scale=args.relocation_scale,
        relocation_fraction=args.relocation_fraction,
        assignment_strategy=args.assignment_strategy,
        selected_scores=selected_scores,
    )
    save_relocation_artifact(
        args.output,
        result=result,
        model_id=args.model_id,
        window_artifact=args.windows,
        selection_artifact=args.selection,
        batch_size=args.batch_size,
        dummy_init=args.dummy_init,
        num_dummy_tokens=args.num_dummy_tokens,
        relocation_scale=args.relocation_scale,
        relocation_fraction=args.relocation_fraction,
        assignment_strategy=args.assignment_strategy,
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
