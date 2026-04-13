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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select BOS neurons by percentile or top-k.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/selections/bos_selection.pt"))
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    if (args.percentile is None) == (args.topk is None):
        raise SystemExit("Specify exactly one of --percentile or --topk")

    scores_payload = load_pt(args.scores)
    scores = scores_payload["tensors"]["scores"]

    if args.percentile is not None:
        selected, threshold = select_neurons_by_percentile(scores, args.percentile)
        method = "percentile"
    else:
        selected, threshold = select_neurons_by_topk(scores, args.topk)
        method = "topk"

    save_selection_artifact(
        args.output,
        selected=selected,
        model_id=args.model_id,
        score_artifact=args.scores,
        method=method,
        percentile=args.percentile,
        topk=args.topk,
        threshold=threshold,
    )
    print(f"Saved {selected.shape[0]} selected neurons to {args.output}")


if __name__ == "__main__":
    main()
