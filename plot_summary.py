from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.artifacts import load_pt
from sink_neurons.env import configure_runtime
from sink_neurons.plotting import (
    plot_attention_relocation,
    plot_bos_score_histogram,
    plot_mean_bos_attention_by_layer,
    plot_mean_token_norm_heatmap,
    plot_neuron_count_vs_percentile,
    plot_norm_relocation,
    plot_relocation_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate summary plots from saved artifacts.")
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--relocation", type=Path, default=None)
    parser.add_argument("--sweep", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    if args.diagnostics is not None:
        payload = load_pt(args.diagnostics)
        plot_mean_token_norm_heatmap(
            payload["tensors"]["mean_token_norms"],
            args.plots_dir / "mean_token_norm_heatmap.png",
        )
        plot_mean_bos_attention_by_layer(
            payload["tensors"]["mean_bos_attention_by_layer"],
            args.plots_dir / "mean_bos_attention_by_layer.png",
        )
    if args.scores is not None:
        payload = load_pt(args.scores)
        plot_bos_score_histogram(payload["tensors"]["scores"], args.plots_dir / "bos_score_histogram.png")
    if args.relocation is not None:
        payload = load_pt(args.relocation)
        tensors = payload["tensors"]
        plot_norm_relocation(
            tensors["bos_norm_before"],
            tensors["bos_norm_after"],
            tensors["dummy_norm_after"],
            args.plots_dir / "norm_relocation.png",
        )
        plot_attention_relocation(
            tensors["bos_attention_before"],
            tensors["bos_attention_after"],
            tensors["dummy_attention_after"],
            args.plots_dir / "attention_relocation.png",
        )
    if args.sweep is not None:
        payload = load_pt(args.sweep)
        percentiles = [float(item["percentile"]) for item in payload["results"]]
        counts = [int(item["count"]) for item in payload["results"]]
        summary = [float(item["summary_metric_mean"]) for item in payload["results"]]
        plot_neuron_count_vs_percentile(percentiles, counts, args.plots_dir / "neuron_count_vs_percentile.png")
        plot_relocation_summary(
            counts,
            summary,
            "Selected Neuron Count",
            args.plots_dir / "relocation_quality_vs_count.png",
        )
        plot_relocation_summary(
            percentiles,
            summary,
            "Percentile Threshold",
            args.plots_dir / "relocation_quality_vs_percentile.png",
        )
    print(f"Saved plots under {args.plots_dir}")


if __name__ == "__main__":
    main()
