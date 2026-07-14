from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.artifacts import load_pt
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID
from sink_neurons.selection import (
    save_selection_artifact,
    select_neurons_by_percentile,
    select_neurons_by_topk,
    select_neurons_by_z_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select BOS neurons by percentile or top-k.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--z-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/selections/bos_selection.pt"))
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    specified_methods = sum(value is not None for value in (args.percentile, args.topk, args.z_threshold))
    if specified_methods != 1:
        raise SystemExit("Specify exactly one of --percentile, --topk, or --z-threshold")

    scores_payload = load_pt(args.scores)
    scores = scores_payload["tensors"]["scores"]

    if args.percentile is not None:
        selected, threshold = select_neurons_by_percentile(scores, args.percentile)
        method = "percentile"
    elif args.topk is not None:
        selected, threshold = select_neurons_by_topk(scores, args.topk)
        method = "topk"
    else:
        z_scores = scores_payload["tensors"].get("z_scores")
        if z_scores is None:
            raise SystemExit("Score artifact does not contain z_scores; regenerate scores first")
        selected, threshold = select_neurons_by_z_score(z_scores, args.z_threshold)
        method = "z_score"

    save_selection_artifact(
        args.output,
        selected=selected,
        model_id=args.model_id,
        score_artifact=args.scores,
        method=method,
        percentile=args.percentile,
        topk=args.topk,
        z_threshold=args.z_threshold,
        threshold=threshold,
    )
    print(f"Saved {selected.shape[0]} selected neurons to {args.output}")


if __name__ == "__main__":
    main()
