from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else ()

from sink_neurons.artifacts import save_artifact
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID, get_mlp_intermediate_size, get_transformer_layers, load_model, load_tokenizer
from sink_neurons.selection import save_selection_artifact, select_neurons_by_topk
from sink_neurons.types import ScoreArtifactMetadata, normalize_path
from sink_neurons.windows import load_windows_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find register-style neurons by scoring MLP activations on discovered sink-token positions. "
            "Defaults to attention-received sinks from 'Attention Sinks: A Catch, Tag, Release Mechanism'."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/register_neurons/scores.pt"))
    parser.add_argument("--selection-dir", type=Path, default=Path("outputs/register_neurons/selections"))
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument(
        "--token-discovery",
        choices=("attention_sink", "high_norm"),
        default="attention_sink",
        help="How to identify token positions before scoring neurons.",
    )
    parser.add_argument(
        "--sink-attention-threshold",
        type=float,
        default=0.2,
        help="Attention-received threshold for attention_sink discovery.",
    )
    parser.add_argument(
        "--min-sink-query-count",
        type=int,
        default=32,
        help="Minimum number of later query tokens required for a key position to be eligible as an attention sink.",
    )
    parser.add_argument(
        "--outlier-layer",
        type=int,
        default=-1,
        help="Layer output used to find high-norm token positions. Negative values count from the final layer.",
    )
    parser.add_argument(
        "--top-layer",
        type=int,
        default=None,
        help="Highest MLP layer index to score. Defaults to the resolved outlier layer.",
    )
    parser.add_argument(
        "--norm-threshold-std",
        type=float,
        default=3.0,
        help="Per-window outlier threshold: mean token norm + this many standard deviations.",
    )
    parser.add_argument(
        "--exclude-positions",
        type=int,
        nargs="*",
        default=[0],
        help="Token positions excluded from outlier detection. Defaults to the synthetic start token at position 0.",
    )
    parser.add_argument(
        "--fallback-top-tokens",
        type=int,
        default=0,
        help="If a window has no threshold outliers, optionally use its top-N norm tokens instead.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.norm_threshold_std < 0:
        raise SystemExit("--norm-threshold-std must be non-negative")
    if args.sink_attention_threshold < 0:
        raise SystemExit("--sink-attention-threshold must be non-negative")
    if args.min_sink_query_count < 1:
        raise SystemExit("--min-sink-query-count must be at least 1")
    if args.fallback_top_tokens < 0:
        raise SystemExit("--fallback-top-tokens must be non-negative")
    if any(topk <= 0 for topk in args.topk):
        raise SystemExit("--topk values must be positive")
    return args


def _resolve_layer_index(layer_idx: int, num_layers: int) -> int:
    resolved = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(f"Layer index {layer_idx} resolves outside [0, {num_layers - 1}]")
    return resolved


def _score_projection_and_activation(mlp: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "act_fn"):
        return mlp.gate_proj, mlp.act_fn
    if hasattr(mlp, "c_fc") and hasattr(mlp, "act"):
        return mlp.c_fc, mlp.act
    if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "act"):
        return mlp.dense_h_to_4h, mlp.act
    if hasattr(mlp, "fc1") and hasattr(mlp, "activation_fn"):
        return mlp.fc1, mlp.activation_fn
    raise AttributeError("Unsupported MLP architecture for register-neuron discovery")


def _iter_slices(num_items: int, batch_size: int):
    for start in range(0, num_items, batch_size):
        yield start, min(start + batch_size, num_items)


def _candidate_mask_like(windows: torch.Tensor, exclude_positions: set[int]) -> torch.Tensor:
    candidate_mask = torch.ones(windows.shape, dtype=torch.bool)
    for position in sorted(exclude_positions):
        if 0 <= position < windows.shape[1]:
            candidate_mask[:, position] = False
    return candidate_mask


@torch.no_grad()
def find_attention_sink_tokens(
    model: torch.nn.Module,
    windows: torch.Tensor,
    *,
    batch_size: int,
    attention_threshold: float,
    min_query_count: int,
    exclude_positions: set[int],
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sink_mask = torch.zeros(windows.shape, dtype=torch.bool)
    max_attention_received = torch.zeros(windows.shape, dtype=torch.float32)
    sink_hit_count = torch.zeros(windows.shape, dtype=torch.long)
    sink_counts = torch.zeros(windows.shape[0], dtype=torch.long)
    candidate_mask = _candidate_mask_like(windows, exclude_positions)

    seq_len = int(windows.shape[1])
    future_query_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32), diagonal=-1)
    denominators = future_query_mask.sum(dim=0).clamp_min(1.0)
    valid_future = denominators >= min_query_count

    for start, end in tqdm(
        _iter_slices(windows.shape[0], batch_size),
        desc="Finding attention sinks",
        total=(windows.shape[0] + batch_size - 1) // batch_size,
        disable=not show_progress,
        dynamic_ncols=True,
    ):
        batch = windows[start:end].to(model.device)
        outputs = model(input_ids=batch, output_attentions=True, use_cache=False, return_dict=True)
        batch_max = torch.zeros((end - start, seq_len), dtype=torch.float32)
        batch_hits = torch.zeros((end - start, seq_len), dtype=torch.long)
        future_mask_device = future_query_mask.to(model.device)
        denominators_device = denominators.to(model.device)

        for attn in outputs.attentions:
            # attn: [batch, heads, query, key]. For each key token, average attention
            # from later causal queries only; this adapts the paper's "attention received"
            # definition to autoregressive windows.
            received = (attn.float() * future_mask_device[None, None, :, :]).sum(dim=2)
            received = received / denominators_device[None, None, :]
            received = received.cpu()
            batch_max = torch.maximum(batch_max, received.amax(dim=1))
            batch_hits += (received >= attention_threshold).sum(dim=1)

        batch_candidate_mask = candidate_mask[start:end] & valid_future[None, :]
        batch_sink_mask = batch_candidate_mask & (batch_hits > 0)
        sink_mask[start:end] = batch_sink_mask
        max_attention_received[start:end] = batch_max
        sink_hit_count[start:end] = batch_hits * batch_candidate_mask.long()
        sink_counts[start:end] = batch_sink_mask.sum(dim=1)

    return sink_mask, max_attention_received, sink_hit_count, sink_counts


@torch.no_grad()
def find_outlier_tokens(
    model: torch.nn.Module,
    windows: torch.Tensor,
    *,
    batch_size: int,
    outlier_layer: int,
    norm_threshold_std: float,
    exclude_positions: set[int],
    fallback_top_tokens: int,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outlier_mask = torch.zeros(windows.shape, dtype=torch.bool)
    token_norms = torch.zeros(windows.shape, dtype=torch.float32)
    thresholds = torch.zeros(windows.shape[0], dtype=torch.float32)
    outlier_counts = torch.zeros(windows.shape[0], dtype=torch.long)

    excluded = torch.tensor(
        [position for position in sorted(exclude_positions) if 0 <= position < windows.shape[1]],
        dtype=torch.long,
    )
    for start, end in tqdm(
        _iter_slices(windows.shape[0], batch_size),
        desc="Finding high-norm tokens",
        total=(windows.shape[0] + batch_size - 1) // batch_size,
        disable=not show_progress,
        dynamic_ncols=True,
    ):
        batch = windows[start:end].to(model.device)
        outputs = model(input_ids=batch, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = outputs.hidden_states[outlier_layer + 1].float()
        norms = torch.linalg.vector_norm(hidden, dim=-1).cpu()

        candidate_mask = torch.ones_like(norms, dtype=torch.bool)
        if excluded.numel() > 0:
            candidate_mask[:, excluded] = False
        candidate_norms = norms.masked_fill(~candidate_mask, float("nan"))
        means = torch.nanmean(candidate_norms, dim=1)
        centered = candidate_norms - means[:, None]
        stds = torch.sqrt(torch.nanmean(centered.square(), dim=1)).clamp_min(1e-8)
        batch_thresholds = means + norm_threshold_std * stds
        batch_mask = candidate_mask & (norms >= batch_thresholds[:, None])

        if fallback_top_tokens > 0:
            empty_rows = batch_mask.sum(dim=1) == 0
            if empty_rows.any():
                fallback_scores = norms.masked_fill(~candidate_mask, float("-inf"))
                top_count = min(fallback_top_tokens, int(candidate_mask.shape[1] - excluded.numel()))
                top_positions = torch.topk(fallback_scores[empty_rows], k=top_count, dim=1).indices
                row_indices = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
                batch_mask[row_indices[:, None], top_positions] = True

        token_norms[start:end] = norms
        thresholds[start:end] = batch_thresholds
        outlier_mask[start:end] = batch_mask
        outlier_counts[start:end] = batch_mask.sum(dim=1)

    return outlier_mask, token_norms, thresholds, outlier_counts


@torch.no_grad()
def score_register_neurons(
    model: torch.nn.Module,
    windows: torch.Tensor,
    outlier_mask: torch.Tensor,
    *,
    batch_size: int,
    top_layer: int,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    intermediate_size = get_mlp_intermediate_size(model)
    activation_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    activation_count_by_layer = torch.zeros(num_layers, dtype=torch.long)
    active_mask: torch.Tensor | None = None
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layers = get_transformer_layers(model)

    def make_hook(layer_idx: int, act_fn: torch.nn.Module):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_mask is None:
                raise RuntimeError("Outlier mask was not set for the current batch")
            gate_act = act_fn(output).float()
            selected = gate_act[active_mask]
            if selected.numel() == 0:
                return
            activation_sum[layer_idx] += selected.sum(dim=0).cpu().double()
            activation_count_by_layer[layer_idx] += int(selected.shape[0])

        return hook

    for layer_idx in range(top_layer + 1):
        projection, act_fn = _score_projection_and_activation(layers[layer_idx].mlp)
        handles.append(projection.register_forward_hook(make_hook(layer_idx, act_fn)))

    try:
        for start, end in tqdm(
            _iter_slices(windows.shape[0], batch_size),
            desc="Scoring register neurons",
            total=(windows.shape[0] + batch_size - 1) // batch_size,
            disable=not show_progress,
            dynamic_ncols=True,
        ):
            active_mask = outlier_mask[start:end].to(model.device)
            batch = windows[start:end].to(model.device)
            _ = model(input_ids=batch, use_cache=False, return_dict=True)
    finally:
        active_mask = None
        for handle in handles:
            handle.remove()

    scores = torch.zeros_like(activation_sum, dtype=torch.float32)
    scored_layers = activation_count_by_layer > 0
    scores[scored_layers] = (activation_sum[scored_layers] / activation_count_by_layer[scored_layers, None]).float()
    return scores, activation_count_by_layer


def _write_summary(
    *,
    csv_path: Path | None,
    md_path: Path | None,
    model_id: str,
    windows_path: Path,
    outlier_layer: int,
    top_layer: int,
    token_discovery: str,
    attention_threshold: float,
    norm_threshold_std: float,
    total_outliers: int,
    outlier_counts: torch.Tensor,
    activation_count_by_layer: torch.Tensor,
    selection_rows: list[dict[str, object]],
    sink_position_rows: list[dict[str, object]],
    sink_token_rows: list[dict[str, object]],
) -> None:
    rows = selection_rows or [{"topk": "", "threshold": "", "selection_artifact": ""}]
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open("w") as handle:
            handle.write("# Register Neuron Discovery\n\n")
            handle.write(f"- model_id: `{model_id}`\n")
            handle.write(f"- windows: `{windows_path.resolve()}`\n")
            handle.write(f"- token_discovery: `{token_discovery}`\n")
            if token_discovery == "attention_sink":
                handle.write(f"- attention_threshold: `{attention_threshold:g}`\n")
            else:
                handle.write(f"- outlier_layer: `{outlier_layer}`\n")
                handle.write(f"- norm_threshold: `mean + {norm_threshold_std:g} std` per window\n")
            handle.write(f"- top_layer: `{top_layer}`\n")
            handle.write(f"- total_selected_tokens: `{total_outliers}`\n")
            handle.write(f"- windows_with_selected_tokens: `{int((outlier_counts > 0).sum().item())}` / `{outlier_counts.numel()}`\n")
            handle.write(
                f"- scored_token_count_per_layer: `{int(activation_count_by_layer[: top_layer + 1].min().item())}`"
                f" to `{int(activation_count_by_layer[: top_layer + 1].max().item())}`\n\n"
            )
            if sink_position_rows:
                handle.write("## Top Sink Positions\n\n")
                handle.write("| position | count |\n")
                handle.write("|---:|---:|\n")
                for row in sink_position_rows:
                    handle.write(f"| {row['position']} | {row['count']} |\n")
                handle.write("\n")
            if sink_token_rows:
                handle.write("## Common Sink Tokens\n\n")
                handle.write("| token | count |\n")
                handle.write("|---|---:|\n")
                for row in sink_token_rows:
                    handle.write(f"| `{row['token']}` | {row['count']} |\n")
                handle.write("\n")
            handle.write("| topk | threshold | selection_artifact |\n")
            handle.write("|---:|---:|---|\n")
            for row in selection_rows:
                handle.write(f"| {row['topk']} | {row['threshold']:.6g} | `{row['selection_artifact']}` |\n")


def _summarize_sink_positions(mask: torch.Tensor, windows: torch.Tensor, model_id: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    positions = mask.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return [], []

    position_counts = Counter(int(position) for _window, position in positions.tolist())
    position_rows = [
        {"position": position, "count": count}
        for position, count in position_counts.most_common(20)
    ]

    try:
        tokenizer = load_tokenizer(model_id)
        token_counts: Counter[str] = Counter()
        for window_idx, position in positions.tolist():
            token_id = int(windows[window_idx, position].item())
            token_counts[repr(tokenizer.decode([token_id]))] += 1
        token_rows = [
            {"token": token, "count": count}
            for token, count in token_counts.most_common(20)
        ]
    except Exception as exc:
        token_rows = [{"token": f"<decode failed: {exc}>", "count": 0}]
    return position_rows, token_rows


def main() -> None:
    configure_runtime()
    args = parse_args()
    windows, _window_payload = load_windows_artifact(args.windows)
    model = load_model(args.model_id, eager_attention=args.token_discovery == "attention_sink")
    num_layers = model.config.num_hidden_layers
    outlier_layer = _resolve_layer_index(args.outlier_layer, num_layers)
    top_layer = outlier_layer if args.top_layer is None else _resolve_layer_index(args.top_layer, num_layers)

    token_norms: torch.Tensor | None = None
    thresholds: torch.Tensor | None = None
    max_attention_received: torch.Tensor | None = None
    sink_hit_count: torch.Tensor | None = None
    if args.token_discovery == "attention_sink":
        outlier_mask, max_attention_received, sink_hit_count, outlier_counts = find_attention_sink_tokens(
            model,
            windows,
            batch_size=args.batch_size,
            attention_threshold=args.sink_attention_threshold,
            min_query_count=args.min_sink_query_count,
            exclude_positions=set(args.exclude_positions),
            show_progress=not args.no_progress,
        )
    else:
        outlier_mask, token_norms, thresholds, outlier_counts = find_outlier_tokens(
            model,
            windows,
            batch_size=args.batch_size,
            outlier_layer=outlier_layer,
            norm_threshold_std=args.norm_threshold_std,
            exclude_positions=set(args.exclude_positions),
            fallback_top_tokens=args.fallback_top_tokens,
            show_progress=not args.no_progress,
        )
    total_outliers = int(outlier_mask.sum().item())
    if total_outliers == 0:
        if args.token_discovery == "attention_sink":
            raise SystemExit("No attention sink token positions were found. Try a lower --sink-attention-threshold.")
        raise SystemExit("No high-norm token positions were found. Try --fallback-top-tokens 1 or a lower threshold.")

    scores, activation_count_by_layer = score_register_neurons(
        model,
        windows,
        outlier_mask,
        batch_size=args.batch_size,
        top_layer=top_layer,
        show_progress=not args.no_progress,
    )

    metadata = ScoreArtifactMetadata(
        model_id=args.model_id,
        score_name="mean_sink_token_gate_activation",
        window_artifact=normalize_path(args.windows),
        num_layers=int(scores.shape[0]),
        intermediate_size=int(scores.shape[1]),
        num_examples=int(windows.shape[0]),
    )
    tensors = {
        "scores": scores,
        "sink_mask": outlier_mask,
        "outlier_mask": outlier_mask,
        "selected_token_counts": outlier_counts,
        "outlier_counts": outlier_counts,
        "activation_count_by_layer": activation_count_by_layer,
    }
    if args.token_discovery == "attention_sink":
        tensors["max_attention_received"] = max_attention_received
        tensors["sink_hit_count"] = sink_hit_count
    else:
        tensors["token_norms"] = token_norms
        tensors["thresholds"] = thresholds

    save_artifact(
        args.output,
        tensors=tensors,
        metadata=metadata,
        config={
            "batch_size": args.batch_size,
            "token_discovery": args.token_discovery,
            "sink_attention_threshold": args.sink_attention_threshold,
            "min_sink_query_count": args.min_sink_query_count,
            "outlier_layer": outlier_layer,
            "top_layer": top_layer,
            "norm_threshold_std": args.norm_threshold_std,
            "exclude_positions": args.exclude_positions,
            "fallback_top_tokens": args.fallback_top_tokens,
            "selection_procedure": (
                "attention-received sink tokens, then mean MLP gate activation on those positions"
                if args.token_discovery == "attention_sink"
                else "per-window high-norm tokens, then mean MLP gate activation on those positions"
            ),
        },
    )

    selection_rows: list[dict[str, object]] = []
    args.selection_dir.mkdir(parents=True, exist_ok=True)
    for topk in args.topk:
        selected, threshold = select_neurons_by_topk(scores[: top_layer + 1], topk)
        selection_path = args.selection_dir / f"selection_register_top{topk}.pt"
        save_selection_artifact(
            selection_path,
            selected=selected,
            model_id=args.model_id,
            score_artifact=args.output,
            method="topk",
            percentile=None,
            topk=topk,
            z_threshold=None,
            threshold=threshold,
        )
        selection_rows.append(
            {
                "topk": topk,
                "threshold": threshold,
                "selection_artifact": str(selection_path.resolve()),
            }
        )

    sink_position_rows, sink_token_rows = _summarize_sink_positions(outlier_mask, windows, args.model_id)
    _write_summary(
        csv_path=args.summary_csv,
        md_path=args.summary_md,
        model_id=args.model_id,
        windows_path=args.windows,
        outlier_layer=outlier_layer,
        top_layer=top_layer,
        token_discovery=args.token_discovery,
        attention_threshold=args.sink_attention_threshold,
        norm_threshold_std=args.norm_threshold_std,
        total_outliers=total_outliers,
        outlier_counts=outlier_counts,
        activation_count_by_layer=activation_count_by_layer,
        selection_rows=selection_rows,
        sink_position_rows=sink_position_rows,
        sink_token_rows=sink_token_rows,
    )
    print(f"Saved register-neuron scores to {args.output}")
    for row in selection_rows:
        print(f"Saved top-{row['topk']} selection to {row['selection_artifact']}")


if __name__ == "__main__":
    main()
