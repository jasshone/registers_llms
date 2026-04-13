from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from sink_neurons.artifacts import load_pt, save_pt
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID, load_model
from sink_neurons.plotting import plot_neuron_count_vs_percentile, plot_relocation_summary
from sink_neurons.relocation import run_relocation_experiment
from sink_neurons.selection import save_selection_artifact, select_neurons_by_percentile
from sink_neurons.windows import load_windows_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep BOS-neuron percentile thresholds.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--percentiles", type=float, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/relocation/percentile_sweep.pt"))
    parser.add_argument("--selection-dir", type=Path, default=Path("outputs/selections/sweep"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    windows, _ = load_windows_artifact(args.windows)
    scores_payload = load_pt(args.scores)
    scores = scores_payload["tensors"]["scores"]
    model = load_model(args.model_id, eager_attention=True)

    selected_counts: list[int] = []
    summary_values: list[float] = []
    results: list[dict[str, object]] = []

    for percentile in tqdm(
        args.percentiles,
        desc="Percentile sweep",
        disable=args.no_progress,
        dynamic_ncols=True,
    ):
        selected, threshold = select_neurons_by_percentile(scores, percentile)
        selection_path = args.selection_dir / f"selection_p{percentile:.5f}.pt"
        save_selection_artifact(
            selection_path,
            selected=selected,
            model_id=args.model_id,
            score_artifact=args.scores,
            method="percentile",
            percentile=percentile,
            topk=None,
            threshold=threshold,
        )
        result = run_relocation_experiment(
            model,
            windows,
            selected,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
        )
        selected_counts.append(int(selected.shape[0]))
        summary_values.append(float(result.summary_metric.mean().item()))
        results.append(
            {
                "percentile": percentile,
                "count": int(selected.shape[0]),
                "threshold": threshold,
                "selection_artifact": str(selection_path.resolve()),
                "summary_metric_mean": float(result.summary_metric.mean().item()),
                "summary_metric_by_layer": result.summary_metric,
                "bos_attention_after": result.bos_attention_after,
                "dummy_attention_after": result.dummy_attention_after,
            }
        )

    payload = {
        "config": {
            "model_id": args.model_id,
            "windows": str(args.windows.resolve()),
            "scores": str(args.scores.resolve()),
            "batch_size": args.batch_size,
            "percentiles": args.percentiles,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pt(args.output, payload)
    plot_neuron_count_vs_percentile(args.percentiles, selected_counts, args.plots_dir / "neuron_count_vs_percentile.png")
    plot_relocation_summary(
        selected_counts,
        summary_values,
        "Selected Neuron Count",
        args.plots_dir / "relocation_quality_vs_count.png",
    )
    plot_relocation_summary(
        args.percentiles,
        summary_values,
        "Percentile Threshold",
        args.plots_dir / "relocation_quality_vs_percentile.png",
    )
    print(f"Saved sweep results to {args.output}")


if __name__ == "__main__":
    main()
