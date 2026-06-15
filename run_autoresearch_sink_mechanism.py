
from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from find_register_neurons import find_attention_sink_tokens
from sink_neurons.artifacts import load_pt, save_pt
from sink_neurons.env import configure_runtime
from sink_neurons.intervention import build_intervened_inputs, _split_mlp_forward
from sink_neurons.modeling import get_transformer_layers, load_model
from sink_neurons.relocation import measure_relocation_baseline
from sink_neurons.windows import build_window_dataloader, load_windows_artifact

OUT = Path('outputs/autoresearch_sink_mechanism')
MAX_WINDOWS_DEEP = 32
BATCH_SIZE = 1
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
TOPKS = [1, 2, 4, 8, 16]
SOURCE_ABS_BONUS = 0.1


@dataclass(frozen=True)
class ModelCfg:
    key: str
    model_id: str
    windows: Path
    score_artifact: Path | None
    reference_selection: Path | None
    baseline_relocation: Path | None
    baseline_summary: str
    use_cached_sink_mask: bool = True


MODELS = [
    ModelCfg(
        key='qwen3_1_7b',
        model_id='Qwen/Qwen3-1.7B',
        windows=Path('outputs/qwen3_1_7b/windows_32x1024.pt'),
        score_artifact=Path('outputs/qwen3_1_7b/attention_sink_register_neurons_percentile_32_include_pos0/scores.pt'),
        reference_selection=Path('outputs/qwen3_1_7b/attention_sink_register_neurons_percentile_32_include_pos0/p9999_layer_subsets/selections/l2_only.pt'),
        baseline_relocation=Path('outputs/qwen3_1_7b/attention_sink_register_neurons_percentile_32_include_pos0/p9999_layer_subsets/relocation/l2_only_s1.pt'),
        baseline_summary='old l2 max-relocation: relocation 0.448906, PPL ratio 1.02591; full l2+27 PPL ratio 0.877612',
    ),
    ModelCfg(
        key='gpt2_medium',
        model_id='gpt2-medium',
        windows=Path('outputs/gpt2_confirmation/windows_128x1023.pt'),
        score_artifact=Path('outputs/gpt2_medium/same_method_scale10_bos_ppl/scores.pt'),
        reference_selection=Path('outputs/gpt2_medium/same_method_scale10_bos_ppl/broad_repair_p99.98000_s10/selections/add1_best7_sel_L00N2176_L00N2725_L01N2776_L03N1346_L04N1558_L05N686_L05N1963_L23N341.pt'),
        baseline_relocation=Path('outputs/gpt2_medium/same_method_scale10_bos_ppl/relocation/selection_p99.98000_s10.pt'),
        baseline_summary='old p99.98 max-relocation: relocation 0.150561, PPL ratio 15.437; repaired set has 7 early + 1 late layer23',
        use_cached_sink_mask=False,
    ),
]


def window_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return (losses * valid).sum(dim=1), valid.sum(dim=1)




def _num_transformer_layers(model: torch.nn.Module) -> int:
    if hasattr(model.config, 'num_hidden_layers'):
        return int(model.config.num_hidden_layers)
    text_config = getattr(model.config, 'text_config', None)
    if text_config is not None and hasattr(text_config, 'num_hidden_layers'):
        return int(text_config.num_hidden_layers)
    if hasattr(model.config, 'n_layer'):
        return int(model.config.n_layer)
    return len(get_transformer_layers(model))

def layer_counts(selected: torch.Tensor) -> str:
    if selected.numel() == 0:
        return ''
    vals = []
    for l in selected[:, 0].unique(sorted=True).tolist():
        vals.append(f'{int(l)}:{int((selected[:, 0] == l).sum())}')
    return ' '.join(vals)


def positive_episodes(bos: torch.Tensor, alpha: float = 0.05) -> list[dict[str, float | int]]:
    slopes = bos[1:] - bos[:-1]
    max_pos = float(slopes.clamp_min(0).max().item()) if slopes.numel() else 0.0
    thresh = alpha * max_pos
    episodes = []
    for idx, s in enumerate(slopes.tolist(), start=1):
        if s > thresh and s > 0:
            episodes.append({'attn_layer': idx, 'prev_mlp_layer': idx - 1, 'slope': float(s), 'threshold': thresh})
    return episodes


def emergence_layers(bos: torch.Tensor, margin: int = 2) -> list[int]:
    """Layers for root-cause search: around first emergence, before late repair confounds."""
    if bos.numel() == 0:
        return [0]
    first = float(bos[0].item())
    mx = float(bos.max().item())
    half = first + 0.5 * (mx - first)
    above = torch.nonzero(bos >= half, as_tuple=False).flatten()
    half_layer = int(above[0].item()) if above.numel() else int(bos.argmax().item())
    slopes = bos[1:] - bos[:-1]
    max_slope_prev = int(slopes.argmax().item()) if slopes.numel() else 0
    lo = max(0, min(half_layer, max_slope_prev + 1) - margin - 1)
    hi = min(int(bos.numel()) - 1, max(half_layer, max_slope_prev + 1) + margin)
    return list(range(lo, hi + 1))


@torch.no_grad()
def clean_baseline(model: torch.nn.Module, windows: torch.Tensor) -> dict[str, torch.Tensor | float | list[dict[str, float | int]]]:
    baseline = measure_relocation_baseline(model, windows, batch_size=BATCH_SIZE, show_progress=False)
    bos = baseline.bos_attention_before.float()
    return {
        'bos_attention': bos,
        'mean_bos': float(bos.mean().item()),
        'max_layer': int(bos.argmax().item()),
        'max_bos': float(bos.max().item()),
        'episodes': positive_episodes(bos),
    }


def load_or_compute_sink_mask(cfg: ModelCfg, model: torch.nn.Module, windows: torch.Tensor, out_dir: Path) -> torch.Tensor:
    if cfg.use_cached_sink_mask and cfg.score_artifact and cfg.score_artifact.exists():
        payload = load_pt(cfg.score_artifact)
        sm = payload.get('tensors', {}).get('sink_mask')
        if sm is not None and tuple(sm.shape) == tuple(load_windows_artifact(cfg.windows)[0].shape):
            return sm[: windows.shape[0]].bool().clone()
    sink_mask, max_received, hit_count, sink_counts = find_attention_sink_tokens(
        model,
        windows,
        batch_size=BATCH_SIZE,
        attention_threshold=0.2,
        min_query_count=32,
        exclude_positions=set(),
        show_progress=False,
    )
    save_pt(out_dir / 'computed_sink_mask.pt', {'tensors': {'sink_mask': sink_mask, 'max_attention_received': max_received, 'sink_hit_count': hit_count, 'sink_counts': sink_counts}})
    return sink_mask.bool()


def make_masks(windows: torch.Tensor, sink_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    bos = torch.zeros_like(sink_mask, dtype=torch.bool)
    bos[:, 0] = True
    return {
        'bos_only': bos,
        'pipeline_sinks': sink_mask.clone(),
        'pipeline_sinks_plus_bos': sink_mask.clone() | bos,
    }


@torch.no_grad()

def _is_opt_decoder_layer(layer: torch.nn.Module) -> bool:
    return all(hasattr(layer, attr) for attr in ("self_attn", "fc1", "fc2", "activation_fn", "final_layer_norm"))


@torch.no_grad()
def _score_candidate_neurons_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    layers: list[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    sums_sink: dict[int, torch.Tensor] = {}
    sums_ctrl: dict[int, torch.Tensor] = {}
    counts_sink: dict[int, int] = {}
    counts_ctrl: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    active_control: torch.Tensor | None = None

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_source is None or active_control is None:
                raise RuntimeError("OPT active masks were not set for current batch")
            layer = get_transformer_layers(model)[layer_idx]
            gate = layer.activation_fn(output).float()
            flat_source = active_source.reshape(-1).to(gate.device).bool()
            flat_control = active_control.reshape(-1).to(gate.device).bool()
            width = int(gate.shape[-1])
            if layer_idx not in sums_sink:
                sums_sink[layer_idx] = torch.zeros(width, dtype=torch.float64)
                sums_ctrl[layer_idx] = torch.zeros(width, dtype=torch.float64)
                counts_sink[layer_idx] = 0
                counts_ctrl[layer_idx] = 0
            if flat_source.any():
                vals = gate[flat_source].detach().cpu().double()
                sums_sink[layer_idx].add_(vals.sum(dim=0))
                counts_sink[layer_idx] += int(vals.shape[0])
            if flat_control.any():
                vals = gate[flat_control].detach().cpu().double()
                sums_ctrl[layer_idx].add_(vals.sum(dim=0))
                counts_ctrl[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer_idx in layers:
        handles.append(transformer_layers[layer_idx].fc1.register_forward_hook(make_hook(layer_idx)))
    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            src = source_mask[offset: offset + bsz].bool()
            ctrl = ~src
            ctrl[:, 0] = False
            active_source = src.to(model.device)
            active_control = ctrl.to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        active_control = None
        for handle in handles:
            handle.remove()

    score_rows = []
    for layer_idx in layers:
        sink_mean = sums_sink[layer_idx] / max(1, counts_sink[layer_idx])
        ctrl_mean = sums_ctrl[layer_idx] / max(1, counts_ctrl[layer_idx])
        gap = sink_mean - ctrl_mean
        combined = gap + SOURCE_ABS_BONUS * sink_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            score_rows.append((float(score), layer_idx, neuron, float(gap[neuron].item()), float(sink_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    score_rows.sort(reverse=True)
    selected_rows = [[layer, neuron] for _score, layer, neuron, *_ in score_rows[:max(TOPKS)]]
    selected = torch.tensor(selected_rows, dtype=torch.long) if selected_rows else torch.empty((0, 2), dtype=torch.long)
    save_pt(out_dir / 'candidate_scores.pt', {'selected_top': selected, 'score_rows': score_rows[:500], 'layers': layers, 'score_type': 'opt_fc1_activation_gap'})
    return {'selected_top': selected, 'layers': torch.tensor(layers, dtype=torch.long)}


@torch.no_grad()
def _estimate_bias_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    for layer_idx in selected[:, 0].unique(sorted=True).tolist():
        neurons = selected[selected[:, 0] == layer_idx][:, 1]
        sums[int(layer_idx)] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[int(layer_idx)] = 0

    def make_hook(layer_idx: int, neurons: torch.Tensor):
        def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_source is None:
                raise RuntimeError("OPT active source mask was not set for current batch")
            layer = get_transformer_layers(model)[layer_idx]
            gate = layer.activation_fn(output).float()
            flat_source = active_source.reshape(-1).to(gate.device).bool()
            if flat_source.any():
                vals = gate[:, neurons.to(gate.device)][flat_source]
                sums[layer_idx].add_(vals.detach().cpu().double().sum(dim=0))
                counts[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer_idx in sums:
        neurons = selected[selected[:, 0] == layer_idx][:, 1]
        handles.append(transformer_layers[layer_idx].fc1.register_forward_hook(make_hook(layer_idx, neurons)))
    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            active_source = source_mask[offset: offset + bsz].bool().to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        for handle in handles:
            handle.remove()
    return {layer_idx: (selected[selected[:, 0] == layer_idx][:, 1].clone(), (sums[layer_idx] / max(1, counts[layer_idx])).float()) for layer_idx in sums}


@contextmanager
def _patched_opt_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    originals: dict[int, Callable] = {}
    transformer_layers = get_transformer_layers(model)
    try:
        for layer_idx, (neurons, bias_values) in bias_by_layer.items():
            layer = transformer_layers[layer_idx]
            originals[layer_idx] = layer.forward

            def make_forward(layer_module: torch.nn.Module, selected_neurons: torch.Tensor, bias: torch.Tensor):
                def forward(
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                    past_key_values=None,
                    use_cache: bool | None = False,
                    position_ids: torch.LongTensor | None = None,
                    **kwargs,
                ) -> torch.Tensor:
                    residual = hidden_states
                    if layer_module.do_layer_norm_before:
                        hidden_states = layer_module.self_attn_layer_norm(hidden_states)
                    hidden_states, _ = layer_module.self_attn(
                        hidden_states=hidden_states,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                    hidden_states = F.dropout(hidden_states, p=layer_module.dropout, training=layer_module.training)
                    hidden_states = residual + hidden_states
                    if not layer_module.do_layer_norm_before:
                        hidden_states = layer_module.self_attn_layer_norm(hidden_states)

                    hidden_states_shape = hidden_states.shape
                    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
                    residual = hidden_states
                    if layer_module.do_layer_norm_before:
                        hidden_states = layer_module.final_layer_norm(hidden_states)

                    gate = layer_module.activation_fn(layer_module.fc1(hidden_states))
                    idx = selected_neurons.to(gate.device)
                    b = bias.to(gate.device, dtype=gate.dtype) * scale
                    batch_size, seq_len = hidden_states_shape[0], hidden_states_shape[1]
                    gate_selected = gate[:, idx].clone()
                    src = source_mask_batch.to(gate.device).bool()
                    expanded = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=gate.device)
                    expanded[:, 0] = src[:, 0]
                    expanded[:, 2:] = src[:, 1:]
                    dummy_pos = 1
                    flat_expanded = expanded.reshape(-1)
                    token_pos = torch.arange(seq_len, device=gate.device).repeat(batch_size)
                    if mode in {'add_only', 'transfer'}:
                        dummy_rows = token_pos == dummy_pos
                        if dummy_rows.any():
                            gate_selected[dummy_rows] = b
                    if mode in {'subtract_only', 'transfer'} and flat_expanded.any():
                        gate_selected[flat_expanded] = gate_selected[flat_expanded] - b
                    gate = gate.clone()
                    gate[:, idx] = gate_selected

                    hidden_states = layer_module.fc2(gate)
                    hidden_states = F.dropout(hidden_states, p=layer_module.dropout, training=layer_module.training)
                    hidden_states = (residual + hidden_states).view(hidden_states_shape)
                    if not layer_module.do_layer_norm_before:
                        hidden_states = layer_module.final_layer_norm(hidden_states)
                    return hidden_states
                return forward

            layer.forward = make_forward(layer, neurons, bias_values)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            transformer_layers[layer_idx].forward = original  # type: ignore[method-assign]

def score_candidate_neurons(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_layers: Iterable[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    layers = sorted({int(l) for l in candidate_layers if 0 <= int(l) < _num_transformer_layers(model)})
    if not layers:
        layers = [0]
    sums_sink: dict[int, torch.Tensor] = {}
    sums_ctrl: dict[int, torch.Tensor] = {}
    counts_sink: dict[int, int] = {}
    counts_ctrl: dict[int, int] = {}
    width_by_layer: dict[int, int] = {}
    handles = []
    offset = {'value': 0}

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0]
            bsz = int(x.shape[0])
            start = offset['value']
            end = start + bsz
            mask = source_mask[start:end].to(x.device).bool()
            control = ~mask
            # avoid padding weirdness by excluding BOS from control unless source explicitly includes it
            control[:, 0] = False
            gate, _ = _split_mlp_forward(module, x)
            width = int(gate.shape[-1])
            if layer_idx not in sums_sink:
                width_by_layer[layer_idx] = width
                sums_sink[layer_idx] = torch.zeros(width, dtype=torch.float64)
                sums_ctrl[layer_idx] = torch.zeros(width, dtype=torch.float64)
                counts_sink[layer_idx] = 0
                counts_ctrl[layer_idx] = 0
            if mask.any():
                vals = gate[mask].float().detach().cpu().double()
                sums_sink[layer_idx].add_(vals.sum(dim=0))
                counts_sink[layer_idx] += int(vals.shape[0])
            if control.any():
                vals = gate[control].float().detach().cpu().double()
                sums_ctrl[layer_idx].add_(vals.sum(dim=0))
                counts_ctrl[layer_idx] += int(vals.shape[0])
        return hook

    for layer in layers:
        handles.append(get_transformer_layers(model)[layer].mlp.register_forward_hook(make_hook(layer)))
    try:
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset['value'] += int(batch.shape[0])
    finally:
        for h in handles:
            h.remove()

    selected_rows = []
    score_rows = []
    for layer in layers:
        sink_mean = sums_sink[layer] / max(1, counts_sink[layer])
        ctrl_mean = sums_ctrl[layer] / max(1, counts_ctrl[layer])
        # positive gap favors features active on source sink tokens.
        gap = sink_mean - ctrl_mean
        # also keep absolute source mean because some sink features are signed through GELU/silu tails.
        combined = gap + SOURCE_ABS_BONUS * sink_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            score_rows.append((float(score), layer, neuron, float(gap[neuron].item()), float(sink_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    score_rows.sort(reverse=True)
    max_top = max(TOPKS)
    for _, layer, neuron, *_ in score_rows[:max_top]:
        selected_rows.append([layer, neuron])
    selected = torch.tensor(selected_rows, dtype=torch.long) if selected_rows else torch.empty((0, 2), dtype=torch.long)
    save_pt(out_dir / 'candidate_scores.pt', {'selected_top': selected, 'score_rows': score_rows[:500], 'layers': layers})
    return {'selected_top': selected, 'layers': torch.tensor(layers, dtype=torch.long)}


@torch.no_grad()
def estimate_bias(model: torch.nn.Module, windows: torch.Tensor, selected: torch.Tensor, source_mask: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    offset = {'value': 0}
    for layer in selected[:, 0].unique(sorted=True).tolist():
        neurons = selected[selected[:, 0] == layer][:, 1]
        sums[int(layer)] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[int(layer)] = 0

    def make_hook(layer_idx: int, neurons: torch.Tensor):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0]
            bsz = int(x.shape[0])
            start = offset['value']
            end = start + bsz
            mask = source_mask[start:end].to(x.device).bool()
            gate, _ = _split_mlp_forward(module, x)
            if mask.any():
                vals = gate[:, :, neurons.to(x.device)][mask]
                sums[layer_idx].add_(vals.float().sum(dim=0).detach().cpu().double())
                counts[layer_idx] += int(vals.shape[0])
        return hook

    for layer in sums:
        neurons = selected[selected[:, 0] == layer][:, 1]
        handles.append(get_transformer_layers(model)[layer].mlp.register_forward_hook(make_hook(layer, neurons)))
    try:
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset['value'] += int(batch.shape[0])
    finally:
        for h in handles:
            h.remove()
    return {layer: (selected[selected[:, 0] == layer][:, 1].clone(), (sums[layer] / max(1, counts[layer])).float()) for layer in sums}


@contextmanager
def patched_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    originals: dict[int, Callable] = {}
    try:
        for layer, (neurons, bias) in bias_by_layer.items():
            mlp = get_transformer_layers(model)[layer].mlp
            originals[layer] = mlp.forward

            def make_forward(layer_mlp: torch.nn.Module, selected_neurons: torch.Tensor, bias_values: torch.Tensor):
                def forward(x: torch.Tensor) -> torch.Tensor:
                    gate, project = _split_mlp_forward(layer_mlp, x)
                    gate = gate.clone()
                    idx = selected_neurons.to(x.device)
                    b = bias_values.to(x.device, dtype=gate.dtype) * scale
                    dummy_pos = 1
                    if mode in {'add_only', 'transfer'}:
                        gate[:, dummy_pos, idx] = b
                    if mode in {'subtract_only', 'transfer'}:
                        mask = source_mask_batch.to(x.device).bool()
                        expanded = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
                        expanded[:, 0] = mask[:, 0]
                        expanded[:, 2:] = mask[:, 1:]
                        expanded[:, dummy_pos] = False
                        gate_sel = gate[..., idx].clone()
                        if expanded.any():
                            gate_sel[expanded] = gate_sel[expanded] - b
                        gate[..., idx] = gate_sel
                    return project(gate)
                return forward
            mlp.forward = make_forward(mlp, neurons, bias)  # type: ignore[method-assign]
        yield
    finally:
        for layer, original in originals.items():
            get_transformer_layers(model)[layer].mlp.forward = original  # type: ignore[method-assign]


@torch.no_grad()
def evaluate_bias_method(
    model: torch.nn.Module,
    windows: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
    mode: str,
    scale: float,
) -> dict[str, float]:
    if selected.numel() == 0:
        return {'dummy_minus_bos_mean': float('-inf'), 'ppl_ratio': float('inf')}
    bias_by_layer = estimate_bias(model, windows, selected, source_mask)
    baseline = measure_relocation_baseline(model, windows, batch_size=BATCH_SIZE, show_progress=False)
    depth = int(baseline.bos_attention_before.numel())
    bos_after = torch.zeros(depth, dtype=torch.float64)
    dummy_after = torch.zeros(depth, dtype=torch.float64)
    clean_loss = 0.0
    changed_loss = 0.0
    tokens = 0
    examples = 0
    offset = 0
    for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
        bsz = int(batch.shape[0])
        src_batch = source_mask[offset: offset + bsz]
        offset += bsz
        batch = batch.to(model.device)
        clean = model(input_ids=batch, use_cache=False, return_dict=True)
        labels = batch.clone()
        labels[:, 0] = -100
        loss, count = window_nll(clean.logits, labels)
        clean_loss += float(loss.sum().item())
        tokens += int(count.sum().item())
        with patched_bias_intervention(model, bias_by_layer, src_batch, scale, mode):
            embeds, mask = build_intervened_inputs(model, batch, dummy_init='zero', num_dummy_tokens=1)
            changed = model(inputs_embeds=embeds, attention_mask=mask, output_attentions=True, use_cache=False, return_dict=True)
        ilabels = torch.full((batch.shape[0], batch.shape[1] + 1), -100, dtype=torch.long, device=batch.device)
        ilabels[:, 2:] = batch[:, 1:]
        iloss, icount = window_nll(changed.logits, ilabels)
        if int(icount.sum().item()) != int(count.sum().item()):
            raise RuntimeError('token mismatch')
        changed_loss += float(iloss.sum().item())
        examples += bsz
        for l, attn in enumerate(changed.attentions):
            bos_after[l] += attn[:, :, 2:, 0].float().mean(dim=(1, 2)).sum().item()
            dummy_after[l] += attn[:, :, 2:, 1].float().mean(dim=(1, 2)).sum().item()
    bos_after = (bos_after / examples).float()
    dummy_after = (dummy_after / examples).float()
    clean_nll = clean_loss / tokens
    changed_nll = changed_loss / tokens
    return {
        'bos_before_mean': float(baseline.bos_attention_before.mean().item()),
        'bos_after_mean': float(bos_after.mean().item()),
        'dummy_after_mean': float(dummy_after.mean().item()),
        'dummy_minus_bos_mean': float((dummy_after - bos_after).mean().item()),
        'baseline_ppl': math.exp(clean_nll),
        'intervened_ppl': math.exp(changed_nll),
        'ppl_ratio': math.exp(changed_nll - clean_nll),
        'delta_mean_nll': changed_nll - clean_nll,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_model(cfg: ModelCfg) -> list[dict]:
    out_dir = OUT / cfg.key
    out_dir.mkdir(parents=True, exist_ok=True)
    windows, _ = load_windows_artifact(cfg.windows)
    windows = windows[:MAX_WINDOWS_DEEP]
    model = load_model(cfg.model_id, eager_attention=True)
    clean = clean_baseline(model, windows)
    sink_mask = load_or_compute_sink_mask(cfg, model, windows, out_dir)
    masks = make_masks(windows, sink_mask)
    episodes = clean['episodes']  # type: ignore[assignment]
    # Root-cause phase: restrict to first-emergence band. Late layers are excluded here
    # because they can be repair/readout features rather than sink-emergence causes.
    expanded_layers = emergence_layers(clean['bos_attention'])  # type: ignore[arg-type]
    if cfg.reference_selection and cfg.reference_selection.exists():
        ref = load_pt(cfg.reference_selection)['tensors']['selected']
        if cfg.key == 'gpt2_medium':
            ref = ref[ref[:, 0] != 23]  # exclude known late repair for first-principles early transfer.
        ref_layers = {int(x) for x in ref[:, 0].tolist()}
        max_early = max(expanded_layers) if expanded_layers else 0
        expanded_layers = sorted(set(expanded_layers) | {l for l in ref_layers if l <= max_early + 1})
    clean_json = {
        'model_id': cfg.model_id,
        'baseline_summary': cfg.baseline_summary,
        'mean_bos_attention': clean['mean_bos'],
        'max_layer': clean['max_layer'],
        'max_bos_attention': clean['max_bos'],
        'episodes': episodes,
        'candidate_layers': expanded_layers,
        'sink_positions': int(sink_mask.sum().item()),
        'windows': int(windows.shape[0]),
    }
    (out_dir / 'clean_mechanism_context.json').write_text(json.dumps(clean_json, indent=2) + '\n')

    rows: list[dict] = []
    for mask_name, mask in masks.items():
        cdir = out_dir / mask_name
        cdir.mkdir(parents=True, exist_ok=True)
        cand = score_candidate_neurons(model, windows, mask, expanded_layers, cdir)['selected_top']
        for topk in TOPKS:
            selected = cand[:topk]
            if selected.numel() == 0:
                continue
            for mode in ['subtract_only', 'add_only', 'transfer']:
                for scale in ([1.0] if mode in {'subtract_only', 'add_only'} else SCALES):
                    try:
                        result = evaluate_bias_method(model, windows, selected, mask, mode, scale)
                        row = {
                            'model_key': cfg.key,
                            'mask': mask_name,
                            'mode': mode,
                            'scale': scale,
                            'topk': int(topk),
                            'neurons': int(selected.shape[0]),
                            'layer_counts': layer_counts(selected),
                            **result,
                        }
                    except Exception as exc:
                        row = {
                            'model_key': cfg.key,
                            'mask': mask_name,
                            'mode': mode,
                            'scale': scale,
                            'topk': int(topk),
                            'neurons': int(selected.shape[0]),
                            'layer_counts': layer_counts(selected),
                            'error': repr(exc),
                        }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    write_csv(out_dir / 'mechanism_sweep.csv', rows)
    save_pt(out_dir / 'mechanism_sweep.pt', {'rows': rows})
    return rows


def summarize(all_rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'summary.csv', all_rows)
    by_model: dict[str, list[dict]] = {}
    for r in all_rows:
        if 'error' in r:
            continue
        by_model.setdefault(str(r['model_key']), []).append(r)
    with (OUT / 'report.md').open('w') as f:
        f.write('# Attention Sink Mechanism Autoresearch\n\n')
        f.write('New first-principles probes: subtract-only necessity, add-only sufficiency, and zero-dummy bias-transfer on candidate pre-spike sink neurons. Existing experiments are used as reference baselines, not rerun.\n\n')
        for model_key, rows in by_model.items():
            f.write(f'## {model_key}\n\n')
            best_transfer = sorted(
                [r for r in rows if r['mode'] == 'transfer'],
                key=lambda r: (-float(r['dummy_minus_bos_mean']), float(r['ppl_ratio']), int(r['neurons'])),
            )[:10]
            best_ppl_transfer = sorted(
                [r for r in rows if r['mode'] == 'transfer' and float(r['dummy_minus_bos_mean']) > 0],
                key=lambda r: (float(r['ppl_ratio']), -float(r['dummy_minus_bos_mean']), int(r['neurons'])),
            )[:10]
            f.write('### Best relocation transfer rows\n\n')
            f.write('| mask | k | scale | layers | dummy-BOS | PPL ratio |\n|---|---:|---:|---|---:|---:|\n')
            for r in best_transfer[:5]:
                f.write(f"| {r['mask']} | {r['topk']} | {float(r['scale']):.2f} | `{r['layer_counts']}` | {float(r['dummy_minus_bos_mean']):.6g} | {float(r['ppl_ratio']):.6g} |\n")
            f.write('\n### Best positive-relocation PPL rows\n\n')
            f.write('| mask | k | scale | layers | dummy-BOS | PPL ratio |\n|---|---:|---:|---|---:|---:|\n')
            for r in best_ppl_transfer[:5]:
                f.write(f"| {r['mask']} | {r['topk']} | {float(r['scale']):.2f} | `{r['layer_counts']}` | {float(r['dummy_minus_bos_mean']):.6g} | {float(r['ppl_ratio']):.6g} |\n")
            f.write('\n')
        f.write('## Provisional mechanism read\n\n')
        f.write('- If add-only creates a split sink but transfer relocates, the sink feature is sufficient for candidacy but source competition matters.\n')
        f.write('- If subtract-only reduces source attention, the selected bias is necessary for sink maintenance.\n')
        f.write('- Prefer actual pipeline sink-token masks when they outperform BOS-only; otherwise BOS-specificity is a model-specific result.\n')


def main() -> None:
    configure_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for cfg in MODELS:
        try:
            all_rows.extend(run_model(cfg))
        except Exception as exc:
            row = {'model_key': cfg.key, 'error': repr(exc)}
            print(json.dumps(row), flush=True)
            all_rows.append(row)
    summarize(all_rows)


if __name__ == '__main__':
    main()


# --- Qwen3 MoE overrides for final split bias transfer ---
# These overrides preserve the dense MLP implementation above while adding the
# flattened expert-neuron convention used by Qwen3 MoE models:
# flat_neuron = expert * moe_intermediate_size + expert_local_neuron.
_dense_score_candidate_neurons = score_candidate_neurons
_dense_estimate_bias = estimate_bias
_dense_patched_bias_intervention = patched_bias_intervention


def _is_qwen3_moe_mlp(mlp: torch.nn.Module) -> bool:
    return hasattr(mlp, 'experts') and hasattr(mlp.experts, 'gate_up_proj') and hasattr(mlp, 'gate')


def _qwen3_moe_intermediate_size(mlp: torch.nn.Module) -> int:
    if hasattr(mlp.experts, 'intermediate_dim'):
        return int(mlp.experts.intermediate_dim)
    if hasattr(mlp, 'config') and hasattr(mlp.config, 'moe_intermediate_size'):
        return int(mlp.config.moe_intermediate_size)
    raise AttributeError('Unsupported Qwen3 MoE architecture: cannot infer expert intermediate size')


def _decode_qwen3_moe_flat(flat_neurons: torch.Tensor, expert_intermediate: int) -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.div(flat_neurons, expert_intermediate, rounding_mode='floor')
    neurons = flat_neurons % expert_intermediate
    return experts.to(dtype=torch.long), neurons.to(dtype=torch.long)


@torch.no_grad()
def _score_candidate_neurons_qwen3_moe(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_layers: Iterable[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    num_layers = _num_transformer_layers(model)
    layers = sorted({int(l) for l in candidate_layers if 0 <= int(l) < num_layers}) or [0]
    first_mlp = get_transformer_layers(model)[layers[0]].mlp
    expert_intermediate = _qwen3_moe_intermediate_size(first_mlp)
    num_experts = int(model.config.num_experts)
    flattened_size = num_experts * expert_intermediate

    sums_sink = {layer: torch.zeros(flattened_size, dtype=torch.float64) for layer in layers}
    sums_ctrl = {layer: torch.zeros(flattened_size, dtype=torch.float64) for layer in layers}
    counts_sink = {layer: 0 for layer in layers}
    counts_ctrl = {layer: 0 for layer in layers}
    active_source: torch.Tensor | None = None
    active_control: torch.Tensor | None = None
    handles = []

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if active_source is None or active_control is None:
                raise RuntimeError('Qwen3 MoE active masks were not set for current batch')
            hidden_states, top_k_index, top_k_weights = inputs
            flat_source = active_source.reshape(-1).to(hidden_states.device).bool()
            flat_control = active_control.reshape(-1).to(hidden_states.device).bool()
            selected_experts = top_k_index
            selected_weights = top_k_weights.float()
            for expert_idx in torch.unique(selected_experts).tolist():
                expert_positions = selected_experts == int(expert_idx)
                if not expert_positions.any():
                    continue
                row_idx, top_pos = torch.where(expert_positions)
                row_source = flat_source[row_idx]
                row_control = flat_control[row_idx]
                if not (row_source.any() or row_control.any()):
                    continue
                current_state = hidden_states[row_idx]
                gate_weight = module.gate_up_proj[int(expert_idx), :expert_intermediate, :]
                gate = torch.nn.functional.linear(current_state, gate_weight)
                gate_act = module.act_fn(gate).float() * selected_weights[row_idx, top_pos, None]
                offset = int(expert_idx) * expert_intermediate
                if row_source.any():
                    vals = gate_act[row_source].detach().cpu().double()
                    sums_sink[layer_idx][offset: offset + expert_intermediate].add_(vals.sum(dim=0))
                    counts_sink[layer_idx] += int(vals.shape[0])
                if row_control.any():
                    vals = gate_act[row_control].detach().cpu().double()
                    sums_ctrl[layer_idx][offset: offset + expert_intermediate].add_(vals.sum(dim=0))
                    counts_ctrl[layer_idx] += int(vals.shape[0])
        return hook

    for layer in layers:
        mlp = get_transformer_layers(model)[layer].mlp
        if not _is_qwen3_moe_mlp(mlp):
            raise AttributeError(f'Layer {layer} is not a supported Qwen3 MoE layer')
        handles.append(mlp.experts.register_forward_hook(make_hook(layer)))

    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            src = source_mask[offset: offset + bsz].bool()
            ctrl = ~src
            ctrl[:, 0] = False
            active_source = src.to(model.device)
            active_control = ctrl.to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        active_control = None
        for handle in handles:
            handle.remove()

    score_rows = []
    for layer in layers:
        sink_mean = sums_sink[layer] / max(1, counts_sink[layer])
        ctrl_mean = sums_ctrl[layer] / max(1, counts_ctrl[layer])
        gap = sink_mean - ctrl_mean
        combined = gap + SOURCE_ABS_BONUS * sink_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            score_rows.append((float(score), layer, neuron, float(gap[neuron].item()), float(sink_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    score_rows.sort(reverse=True)
    selected_rows = [[layer, neuron] for _score, layer, neuron, *_ in score_rows[:max(TOPKS)]]
    selected = torch.tensor(selected_rows, dtype=torch.long) if selected_rows else torch.empty((0, 2), dtype=torch.long)
    save_pt(out_dir / 'candidate_scores.pt', {
        'selected_top': selected,
        'score_rows': score_rows[:500],
        'layers': layers,
        'score_type': 'qwen3_moe_router_weighted_gate_gap',
        'num_experts': num_experts,
        'moe_intermediate_size': expert_intermediate,
        'flattened_neuron_convention': 'expert * moe_intermediate_size + neuron',
    })
    return {'selected_top': selected, 'layers': torch.tensor(layers, dtype=torch.long)}



def _is_opt_decoder_layer(layer: torch.nn.Module) -> bool:
    return all(hasattr(layer, attr) for attr in ("self_attn", "fc1", "fc2", "activation_fn", "final_layer_norm"))


@torch.no_grad()
def _score_candidate_neurons_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    layers: list[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    sums_sink: dict[int, torch.Tensor] = {}
    sums_ctrl: dict[int, torch.Tensor] = {}
    counts_sink: dict[int, int] = {}
    counts_ctrl: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    active_control: torch.Tensor | None = None

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_source is None or active_control is None:
                raise RuntimeError("OPT active masks were not set for current batch")
            layer = get_transformer_layers(model)[layer_idx]
            gate = layer.activation_fn(output).float()
            flat_source = active_source.reshape(-1).to(gate.device).bool()
            flat_control = active_control.reshape(-1).to(gate.device).bool()
            width = int(gate.shape[-1])
            if layer_idx not in sums_sink:
                sums_sink[layer_idx] = torch.zeros(width, dtype=torch.float64)
                sums_ctrl[layer_idx] = torch.zeros(width, dtype=torch.float64)
                counts_sink[layer_idx] = 0
                counts_ctrl[layer_idx] = 0
            if flat_source.any():
                vals = gate[flat_source].detach().cpu().double()
                sums_sink[layer_idx].add_(vals.sum(dim=0))
                counts_sink[layer_idx] += int(vals.shape[0])
            if flat_control.any():
                vals = gate[flat_control].detach().cpu().double()
                sums_ctrl[layer_idx].add_(vals.sum(dim=0))
                counts_ctrl[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer_idx in layers:
        handles.append(transformer_layers[layer_idx].fc1.register_forward_hook(make_hook(layer_idx)))
    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            src = source_mask[offset: offset + bsz].bool()
            ctrl = ~src
            ctrl[:, 0] = False
            active_source = src.to(model.device)
            active_control = ctrl.to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        active_control = None
        for handle in handles:
            handle.remove()

    score_rows = []
    for layer_idx in layers:
        sink_mean = sums_sink[layer_idx] / max(1, counts_sink[layer_idx])
        ctrl_mean = sums_ctrl[layer_idx] / max(1, counts_ctrl[layer_idx])
        gap = sink_mean - ctrl_mean
        combined = gap + SOURCE_ABS_BONUS * sink_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            score_rows.append((float(score), layer_idx, neuron, float(gap[neuron].item()), float(sink_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    score_rows.sort(reverse=True)
    selected_rows = [[layer, neuron] for _score, layer, neuron, *_ in score_rows[:max(TOPKS)]]
    selected = torch.tensor(selected_rows, dtype=torch.long) if selected_rows else torch.empty((0, 2), dtype=torch.long)
    save_pt(out_dir / 'candidate_scores.pt', {'selected_top': selected, 'score_rows': score_rows[:500], 'layers': layers, 'score_type': 'opt_fc1_activation_gap'})
    return {'selected_top': selected, 'layers': torch.tensor(layers, dtype=torch.long)}


@torch.no_grad()
def _estimate_bias_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    for layer_idx in selected[:, 0].unique(sorted=True).tolist():
        neurons = selected[selected[:, 0] == layer_idx][:, 1]
        sums[int(layer_idx)] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[int(layer_idx)] = 0

    def make_hook(layer_idx: int, neurons: torch.Tensor):
        def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_source is None:
                raise RuntimeError("OPT active source mask was not set for current batch")
            layer = get_transformer_layers(model)[layer_idx]
            gate = layer.activation_fn(output).float()
            flat_source = active_source.reshape(-1).to(gate.device).bool()
            if flat_source.any():
                vals = gate[:, neurons.to(gate.device)][flat_source]
                sums[layer_idx].add_(vals.detach().cpu().double().sum(dim=0))
                counts[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer_idx in sums:
        neurons = selected[selected[:, 0] == layer_idx][:, 1]
        handles.append(transformer_layers[layer_idx].fc1.register_forward_hook(make_hook(layer_idx, neurons)))
    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            active_source = source_mask[offset: offset + bsz].bool().to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        for handle in handles:
            handle.remove()
    return {layer_idx: (selected[selected[:, 0] == layer_idx][:, 1].clone(), (sums[layer_idx] / max(1, counts[layer_idx])).float()) for layer_idx in sums}


@contextmanager
def _patched_opt_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    originals: dict[int, Callable] = {}
    transformer_layers = get_transformer_layers(model)
    try:
        for layer_idx, (neurons, bias_values) in bias_by_layer.items():
            layer = transformer_layers[layer_idx]
            originals[layer_idx] = layer.forward

            def make_forward(layer_module: torch.nn.Module, selected_neurons: torch.Tensor, bias: torch.Tensor):
                def forward(
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                    past_key_values=None,
                    use_cache: bool | None = False,
                    position_ids: torch.LongTensor | None = None,
                    **kwargs,
                ) -> torch.Tensor:
                    residual = hidden_states
                    if layer_module.do_layer_norm_before:
                        hidden_states = layer_module.self_attn_layer_norm(hidden_states)
                    hidden_states, _ = layer_module.self_attn(
                        hidden_states=hidden_states,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                    hidden_states = F.dropout(hidden_states, p=layer_module.dropout, training=layer_module.training)
                    hidden_states = residual + hidden_states
                    if not layer_module.do_layer_norm_before:
                        hidden_states = layer_module.self_attn_layer_norm(hidden_states)

                    hidden_states_shape = hidden_states.shape
                    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
                    residual = hidden_states
                    if layer_module.do_layer_norm_before:
                        hidden_states = layer_module.final_layer_norm(hidden_states)

                    gate = layer_module.activation_fn(layer_module.fc1(hidden_states))
                    idx = selected_neurons.to(gate.device)
                    b = bias.to(gate.device, dtype=gate.dtype) * scale
                    batch_size, seq_len = hidden_states_shape[0], hidden_states_shape[1]
                    gate_selected = gate[:, idx].clone()
                    src = source_mask_batch.to(gate.device).bool()
                    expanded = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=gate.device)
                    expanded[:, 0] = src[:, 0]
                    expanded[:, 2:] = src[:, 1:]
                    dummy_pos = 1
                    flat_expanded = expanded.reshape(-1)
                    token_pos = torch.arange(seq_len, device=gate.device).repeat(batch_size)
                    if mode in {'add_only', 'transfer'}:
                        dummy_rows = token_pos == dummy_pos
                        if dummy_rows.any():
                            gate_selected[dummy_rows] = b
                    if mode in {'subtract_only', 'transfer'} and flat_expanded.any():
                        gate_selected[flat_expanded] = gate_selected[flat_expanded] - b
                    gate = gate.clone()
                    gate[:, idx] = gate_selected

                    hidden_states = layer_module.fc2(gate)
                    hidden_states = F.dropout(hidden_states, p=layer_module.dropout, training=layer_module.training)
                    hidden_states = (residual + hidden_states).view(hidden_states_shape)
                    if not layer_module.do_layer_norm_before:
                        hidden_states = layer_module.final_layer_norm(hidden_states)
                    return hidden_states
                return forward

            layer.forward = make_forward(layer, neurons, bias_values)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            transformer_layers[layer_idx].forward = original  # type: ignore[method-assign]

def score_candidate_neurons(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_layers: Iterable[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    layers = list(candidate_layers)
    probe_layer = layers[0] if layers else 0
    probe = get_transformer_layers(model)[int(probe_layer)]
    if _is_opt_decoder_layer(probe):
        return _score_candidate_neurons_opt(model, windows, source_mask, layers, out_dir)
    mlp = probe.mlp
    if _is_qwen3_moe_mlp(mlp):
        return _score_candidate_neurons_qwen3_moe(model, windows, source_mask, layers, out_dir)
    return _dense_score_candidate_neurons(model, windows, source_mask, layers, out_dir)


@torch.no_grad()
def _estimate_bias_qwen3_moe(
    model: torch.nn.Module,
    windows: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    for layer in selected[:, 0].unique(sorted=True).tolist():
        neurons = selected[selected[:, 0] == layer][:, 1]
        sums[int(layer)] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[int(layer)] = 0

    def make_hook(layer_idx: int, flat_neurons: torch.Tensor):
        mlp = get_transformer_layers(model)[layer_idx].mlp
        expert_intermediate = _qwen3_moe_intermediate_size(mlp)
        selected_experts, selected_neurons = _decode_qwen3_moe_flat(flat_neurons, expert_intermediate)
        by_expert: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for expert_idx in torch.unique(selected_experts).tolist():
            mask = selected_experts == int(expert_idx)
            selected_positions = torch.nonzero(mask, as_tuple=False).squeeze(1).to(dtype=torch.long)
            by_expert[int(expert_idx)] = (selected_neurons[mask].to(dtype=torch.long), selected_positions)

        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if active_source is None:
                raise RuntimeError('Qwen3 MoE active source mask was not set for current batch')
            hidden_states, top_k_index, _top_k_weights = inputs
            flat_source = active_source.reshape(-1).to(hidden_states.device).bool()
            selected_route_experts = top_k_index
            for expert_idx, (local_neurons, selected_positions) in by_expert.items():
                expert_positions = selected_route_experts == int(expert_idx)
                if not expert_positions.any():
                    continue
                row_idx, _top_pos = torch.where(expert_positions)
                row_source = flat_source[row_idx]
                if not row_source.any():
                    continue
                current_state = hidden_states[row_idx[row_source]]
                gate_weight = module.gate_up_proj[int(expert_idx), :expert_intermediate, :]
                gate = torch.nn.functional.linear(current_state, gate_weight)
                gate_act = module.act_fn(gate).float()
                vals = gate_act[:, local_neurons.to(gate_act.device)].detach().cpu().double()
                sums[layer_idx][selected_positions] += vals.sum(dim=0)
                counts[layer_idx] += int(vals.shape[0])
        return hook

    for layer in sums:
        neurons = selected[selected[:, 0] == layer][:, 1]
        mlp = get_transformer_layers(model)[layer].mlp
        handles.append(mlp.experts.register_forward_hook(make_hook(layer, neurons)))

    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            active_source = source_mask[offset: offset + bsz].bool().to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        for handle in handles:
            handle.remove()
    return {layer: (selected[selected[:, 0] == layer][:, 1].clone(), (sums[layer] / max(1, counts[layer])).float()) for layer in sums}


def estimate_bias(model: torch.nn.Module, windows: torch.Tensor, selected: torch.Tensor, source_mask: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if selected.numel() == 0:
        return {}
    first_layer = int(selected[0, 0].item())
    probe = get_transformer_layers(model)[first_layer]
    if _is_opt_decoder_layer(probe):
        return _estimate_bias_opt(model, windows, selected, source_mask)
    mlp = probe.mlp
    if _is_qwen3_moe_mlp(mlp):
        return _estimate_bias_qwen3_moe(model, windows, selected, source_mask)
    return _dense_estimate_bias(model, windows, selected, source_mask)


@contextmanager
def _patched_qwen3_moe_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    originals: dict[int, Callable] = {}
    try:
        for layer, (flat_neurons, bias_values) in bias_by_layer.items():
            mlp = get_transformer_layers(model)[layer].mlp
            originals[layer] = mlp.forward
            expert_intermediate = _qwen3_moe_intermediate_size(mlp)
            selected_experts, selected_neurons = _decode_qwen3_moe_flat(flat_neurons, expert_intermediate)
            by_expert: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for expert_idx in torch.unique(selected_experts).tolist():
                mask = selected_experts == int(expert_idx)
                by_expert[int(expert_idx)] = (
                    selected_neurons[mask].to(dtype=torch.long),
                    torch.nonzero(mask, as_tuple=False).squeeze(1).to(dtype=torch.long),
                    bias_values[mask].float(),
                )

            def make_forward(layer_mlp: torch.nn.Module, expert_map: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
                experts_module = layer_mlp.experts

                def forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
                    _, routing_weights, selected_route_experts = layer_mlp.gate(hidden_states_reshaped)
                    final_hidden_states = torch.zeros_like(hidden_states_reshaped)
                    expert_mask = torch.nn.functional.one_hot(selected_route_experts, num_classes=experts_module.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                    src = source_mask_batch.to(hidden_states.device).bool()
                    expanded = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=hidden_states.device)
                    expanded[:, 0] = src[:, 0]
                    expanded[:, 2:] = src[:, 1:]
                    dummy_pos = 1
                    flat_source = expanded.reshape(-1)

                    for expert_hit_idx in expert_hit:
                        expert_idx = int(expert_hit_idx[0].item())
                        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                        current_state = hidden_states_reshaped[token_idx]
                        gate_up = torch.nn.functional.linear(current_state, experts_module.gate_up_proj[expert_idx])
                        gate, up = gate_up.chunk(2, dim=-1)
                        gate_act = experts_module.act_fn(gate)

                        if expert_idx in expert_map:
                            local_neurons, _positions, local_bias = expert_map[expert_idx]
                            local_neurons = local_neurons.to(gate_act.device)
                            b = (local_bias.to(gate_act.device, dtype=gate_act.dtype) * scale)
                            gate_selected = gate_act[:, local_neurons].clone()
                            token_pos = token_idx % sequence_length
                            source_rows = flat_source[token_idx]
                            if mode in {'add_only', 'transfer'}:
                                dummy_rows = token_pos == dummy_pos
                                if dummy_rows.any():
                                    gate_selected[dummy_rows] = b
                            if mode in {'subtract_only', 'transfer'} and source_rows.any():
                                gate_selected[source_rows] = gate_selected[source_rows] - b
                            gate_act = gate_act.clone()
                            gate_act[:, local_neurons] = gate_selected

                        current_hidden_states = gate_act * up
                        current_hidden_states = torch.nn.functional.linear(current_hidden_states, experts_module.down_proj[expert_idx])
                        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
                        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
                    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

                return forward

            mlp.forward = make_forward(mlp, by_expert)  # type: ignore[method-assign]
        yield
    finally:
        for layer, original in originals.items():
            get_transformer_layers(model)[layer].mlp.forward = original  # type: ignore[method-assign]


@contextmanager
def patched_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    if bias_by_layer:
        first_layer = next(iter(bias_by_layer))
        probe = get_transformer_layers(model)[first_layer]
        if _is_opt_decoder_layer(probe):
            with _patched_opt_bias_intervention(model, bias_by_layer, source_mask_batch, scale, mode):
                yield
            return
        mlp = probe.mlp
        if _is_qwen3_moe_mlp(mlp):
            with _patched_qwen3_moe_bias_intervention(model, bias_by_layer, source_mask_batch, scale, mode):
                yield
            return
    with _dense_patched_bias_intervention(model, bias_by_layer, source_mask_batch, scale, mode):
        yield


# --- OPT decoder-layer overrides for final split bias transfer ---
# OPT places fc1/activation/fc2 directly on the decoder layer rather than under
# layer.mlp. Use fc2's input as the post-activation MLP feature tensor.
_prev_score_candidate_neurons = score_candidate_neurons
_prev_estimate_bias = estimate_bias
_prev_patched_bias_intervention = patched_bias_intervention


def _is_opt_decoder_layer(layer: torch.nn.Module) -> bool:
    return all(hasattr(layer, attr) for attr in ('fc1', 'fc2', 'activation_fn')) and not hasattr(layer, 'mlp')


def _uses_opt_decoder_layers(model: torch.nn.Module, layers: Iterable[int] | None = None) -> bool:
    transformer_layers = get_transformer_layers(model)
    if layers is None:
        layers = range(len(transformer_layers))
    return any(_is_opt_decoder_layer(transformer_layers[int(layer)]) for layer in layers)


@torch.no_grad()
def _score_candidate_neurons_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_layers: Iterable[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    max_layers = _num_transformer_layers(model)
    layers = sorted({int(l) for l in candidate_layers if 0 <= int(l) < max_layers})
    if not layers:
        layers = [0]
    sums_sink: dict[int, torch.Tensor] = {}
    sums_ctrl: dict[int, torch.Tensor] = {}
    counts_sink: dict[int, int] = {}
    counts_ctrl: dict[int, int] = {}
    handles = []
    offset = {'value': 0}
    seq_len = int(source_mask.shape[1])

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            gate = inputs[0]
            width = int(gate.shape[-1])
            bsz = int(gate.shape[0]) // seq_len
            if bsz <= 0:
                return
            start = offset['value']
            end = start + bsz
            mask = source_mask[start:end].to(gate.device).bool().reshape(-1)
            control = ~mask.view(bsz, seq_len)
            control[:, 0] = False
            control = control.reshape(-1)
            gate2 = gate.view(bsz * seq_len, width)
            if layer_idx not in sums_sink:
                sums_sink[layer_idx] = torch.zeros(width, dtype=torch.float64)
                sums_ctrl[layer_idx] = torch.zeros(width, dtype=torch.float64)
                counts_sink[layer_idx] = 0
                counts_ctrl[layer_idx] = 0
            if mask.any():
                vals = gate2[mask].float().detach().cpu().double()
                sums_sink[layer_idx].add_(vals.sum(dim=0))
                counts_sink[layer_idx] += int(vals.shape[0])
            if control.any():
                vals = gate2[control].float().detach().cpu().double()
                sums_ctrl[layer_idx].add_(vals.sum(dim=0))
                counts_ctrl[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer in layers:
        handles.append(transformer_layers[layer].fc2.register_forward_hook(make_hook(layer)))
    try:
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset['value'] += int(batch.shape[0])
    finally:
        for h in handles:
            h.remove()

    selected_rows = []
    score_rows = []
    for layer in layers:
        if layer not in sums_sink:
            continue
        sink_mean = sums_sink[layer] / max(1, counts_sink[layer])
        ctrl_mean = sums_ctrl[layer] / max(1, counts_ctrl[layer])
        gap = sink_mean - ctrl_mean
        combined = gap + SOURCE_ABS_BONUS * sink_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            score_rows.append((float(score), layer, neuron, float(gap[neuron].item()), float(sink_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    score_rows.sort(reverse=True)
    for _, layer, neuron, *_ in score_rows[:max(TOPKS)]:
        selected_rows.append([layer, neuron])
    selected = torch.tensor(selected_rows, dtype=torch.long) if selected_rows else torch.empty((0, 2), dtype=torch.long)
    save_pt(out_dir / 'candidate_scores.pt', {'selected_top': selected, 'score_rows': score_rows[:500], 'layers': layers, 'score_type': 'opt_fc2_input_source_control_gap'})
    return {'selected_top': selected, 'layers': torch.tensor(layers, dtype=torch.long)}


@torch.no_grad()
def _estimate_bias_opt(model: torch.nn.Module, windows: torch.Tensor, selected: torch.Tensor, source_mask: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    offset = {'value': 0}
    seq_len = int(source_mask.shape[1])
    for layer in selected[:, 0].unique(sorted=True).tolist():
        neurons = selected[selected[:, 0] == layer][:, 1]
        sums[int(layer)] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[int(layer)] = 0

    def make_hook(layer_idx: int, neurons: torch.Tensor):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            gate = inputs[0]
            width = int(gate.shape[-1])
            bsz = int(gate.shape[0]) // seq_len
            if bsz <= 0:
                return
            start = offset['value']
            end = start + bsz
            mask = source_mask[start:end].to(gate.device).bool().reshape(-1)
            gate2 = gate.view(bsz * seq_len, width)
            if mask.any():
                vals = gate2[:, neurons.to(gate.device)][mask]
                sums[layer_idx].add_(vals.float().sum(dim=0).detach().cpu().double())
                counts[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer in sums:
        neurons = selected[selected[:, 0] == layer][:, 1]
        handles.append(transformer_layers[layer].fc2.register_forward_hook(make_hook(layer, neurons)))
    try:
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset['value'] += int(batch.shape[0])
    finally:
        for h in handles:
            h.remove()
    return {layer: (selected[selected[:, 0] == layer][:, 1].clone(), (sums[layer] / max(1, counts[layer])).float()) for layer in sums}


@contextmanager
def _patched_opt_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    originals: dict[int, Callable] = {}
    try:
        transformer_layers = get_transformer_layers(model)
        for layer, (neurons, bias) in bias_by_layer.items():
            fc2 = transformer_layers[layer].fc2
            originals[int(layer)] = fc2.forward

            def make_forward(original: Callable, selected_neurons: torch.Tensor, bias_values: torch.Tensor):
                def forward(gate: torch.Tensor) -> torch.Tensor:
                    gate = gate.clone()
                    idx = selected_neurons.to(gate.device)
                    b = bias_values.to(gate.device, dtype=gate.dtype) * scale
                    bsz = int(source_mask_batch.shape[0])
                    seq_len = int(gate.shape[0]) // bsz
                    gate3 = gate.view(bsz, seq_len, gate.shape[-1])
                    dummy_pos = 1
                    if mode in {'add_only', 'transfer'}:
                        gate3[:, dummy_pos, idx] = b
                    if mode in {'subtract_only', 'transfer'}:
                        mask = source_mask_batch.to(gate.device).bool()
                        expanded = torch.zeros((bsz, seq_len), dtype=torch.bool, device=gate.device)
                        expanded[:, 0] = mask[:, 0]
                        expanded[:, 2:] = mask[:, 1:]
                        expanded[:, dummy_pos] = False
                        gate_sel = gate3[..., idx].clone()
                        if expanded.any():
                            gate_sel[expanded] = gate_sel[expanded] - b
                        gate3[..., idx] = gate_sel
                    return original(gate3.view_as(gate))
                return forward
            fc2.forward = make_forward(fc2.forward, neurons, bias)  # type: ignore[method-assign]
        yield
    finally:
        transformer_layers = get_transformer_layers(model)
        for layer, original in originals.items():
            transformer_layers[layer].fc2.forward = original  # type: ignore[method-assign]



def _is_opt_decoder_layer(layer: torch.nn.Module) -> bool:
    return all(hasattr(layer, attr) for attr in ("self_attn", "fc1", "fc2", "activation_fn", "final_layer_norm"))


@torch.no_grad()
def _score_candidate_neurons_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    layers: list[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    sums_sink: dict[int, torch.Tensor] = {}
    sums_ctrl: dict[int, torch.Tensor] = {}
    counts_sink: dict[int, int] = {}
    counts_ctrl: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    active_control: torch.Tensor | None = None

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_source is None or active_control is None:
                raise RuntimeError("OPT active masks were not set for current batch")
            layer = get_transformer_layers(model)[layer_idx]
            gate = layer.activation_fn(output).float()
            flat_source = active_source.reshape(-1).to(gate.device).bool()
            flat_control = active_control.reshape(-1).to(gate.device).bool()
            width = int(gate.shape[-1])
            if layer_idx not in sums_sink:
                sums_sink[layer_idx] = torch.zeros(width, dtype=torch.float64)
                sums_ctrl[layer_idx] = torch.zeros(width, dtype=torch.float64)
                counts_sink[layer_idx] = 0
                counts_ctrl[layer_idx] = 0
            if flat_source.any():
                vals = gate[flat_source].detach().cpu().double()
                sums_sink[layer_idx].add_(vals.sum(dim=0))
                counts_sink[layer_idx] += int(vals.shape[0])
            if flat_control.any():
                vals = gate[flat_control].detach().cpu().double()
                sums_ctrl[layer_idx].add_(vals.sum(dim=0))
                counts_ctrl[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer_idx in layers:
        handles.append(transformer_layers[layer_idx].fc1.register_forward_hook(make_hook(layer_idx)))
    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            src = source_mask[offset: offset + bsz].bool()
            ctrl = ~src
            ctrl[:, 0] = False
            active_source = src.to(model.device)
            active_control = ctrl.to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        active_control = None
        for handle in handles:
            handle.remove()

    score_rows = []
    for layer_idx in layers:
        sink_mean = sums_sink[layer_idx] / max(1, counts_sink[layer_idx])
        ctrl_mean = sums_ctrl[layer_idx] / max(1, counts_ctrl[layer_idx])
        gap = sink_mean - ctrl_mean
        combined = gap + SOURCE_ABS_BONUS * sink_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            score_rows.append((float(score), layer_idx, neuron, float(gap[neuron].item()), float(sink_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    score_rows.sort(reverse=True)
    selected_rows = [[layer, neuron] for _score, layer, neuron, *_ in score_rows[:max(TOPKS)]]
    selected = torch.tensor(selected_rows, dtype=torch.long) if selected_rows else torch.empty((0, 2), dtype=torch.long)
    save_pt(out_dir / 'candidate_scores.pt', {'selected_top': selected, 'score_rows': score_rows[:500], 'layers': layers, 'score_type': 'opt_fc1_activation_gap'})
    return {'selected_top': selected, 'layers': torch.tensor(layers, dtype=torch.long)}


@torch.no_grad()
def _estimate_bias_opt(
    model: torch.nn.Module,
    windows: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    active_source: torch.Tensor | None = None
    for layer_idx in selected[:, 0].unique(sorted=True).tolist():
        neurons = selected[selected[:, 0] == layer_idx][:, 1]
        sums[int(layer_idx)] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[int(layer_idx)] = 0

    def make_hook(layer_idx: int, neurons: torch.Tensor):
        def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if active_source is None:
                raise RuntimeError("OPT active source mask was not set for current batch")
            layer = get_transformer_layers(model)[layer_idx]
            gate = layer.activation_fn(output).float()
            flat_source = active_source.reshape(-1).to(gate.device).bool()
            if flat_source.any():
                vals = gate[:, neurons.to(gate.device)][flat_source]
                sums[layer_idx].add_(vals.detach().cpu().double().sum(dim=0))
                counts[layer_idx] += int(vals.shape[0])
        return hook

    transformer_layers = get_transformer_layers(model)
    for layer_idx in sums:
        neurons = selected[selected[:, 0] == layer_idx][:, 1]
        handles.append(transformer_layers[layer_idx].fc1.register_forward_hook(make_hook(layer_idx, neurons)))
    try:
        offset = 0
        for batch in build_window_dataloader(windows, batch_size=BATCH_SIZE):
            bsz = int(batch.shape[0])
            active_source = source_mask[offset: offset + bsz].bool().to(model.device)
            model(input_ids=batch.to(model.device), use_cache=False, return_dict=True)
            offset += bsz
    finally:
        active_source = None
        for handle in handles:
            handle.remove()
    return {layer_idx: (selected[selected[:, 0] == layer_idx][:, 1].clone(), (sums[layer_idx] / max(1, counts[layer_idx])).float()) for layer_idx in sums}


@contextmanager
def _patched_opt_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    originals: dict[int, Callable] = {}
    transformer_layers = get_transformer_layers(model)
    try:
        for layer_idx, (neurons, bias_values) in bias_by_layer.items():
            layer = transformer_layers[layer_idx]
            originals[layer_idx] = layer.forward

            def make_forward(layer_module: torch.nn.Module, selected_neurons: torch.Tensor, bias: torch.Tensor):
                def forward(
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                    past_key_values=None,
                    use_cache: bool | None = False,
                    position_ids: torch.LongTensor | None = None,
                    **kwargs,
                ) -> torch.Tensor:
                    residual = hidden_states
                    if layer_module.do_layer_norm_before:
                        hidden_states = layer_module.self_attn_layer_norm(hidden_states)
                    hidden_states, _ = layer_module.self_attn(
                        hidden_states=hidden_states,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                    hidden_states = F.dropout(hidden_states, p=layer_module.dropout, training=layer_module.training)
                    hidden_states = residual + hidden_states
                    if not layer_module.do_layer_norm_before:
                        hidden_states = layer_module.self_attn_layer_norm(hidden_states)

                    hidden_states_shape = hidden_states.shape
                    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
                    residual = hidden_states
                    if layer_module.do_layer_norm_before:
                        hidden_states = layer_module.final_layer_norm(hidden_states)

                    gate = layer_module.activation_fn(layer_module.fc1(hidden_states))
                    idx = selected_neurons.to(gate.device)
                    b = bias.to(gate.device, dtype=gate.dtype) * scale
                    batch_size, seq_len = hidden_states_shape[0], hidden_states_shape[1]
                    gate_selected = gate[:, idx].clone()
                    src = source_mask_batch.to(gate.device).bool()
                    expanded = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=gate.device)
                    expanded[:, 0] = src[:, 0]
                    expanded[:, 2:] = src[:, 1:]
                    dummy_pos = 1
                    flat_expanded = expanded.reshape(-1)
                    token_pos = torch.arange(seq_len, device=gate.device).repeat(batch_size)
                    if mode in {'add_only', 'transfer'}:
                        dummy_rows = token_pos == dummy_pos
                        if dummy_rows.any():
                            gate_selected[dummy_rows] = b
                    if mode in {'subtract_only', 'transfer'} and flat_expanded.any():
                        gate_selected[flat_expanded] = gate_selected[flat_expanded] - b
                    gate = gate.clone()
                    gate[:, idx] = gate_selected

                    hidden_states = layer_module.fc2(gate)
                    hidden_states = F.dropout(hidden_states, p=layer_module.dropout, training=layer_module.training)
                    hidden_states = (residual + hidden_states).view(hidden_states_shape)
                    if not layer_module.do_layer_norm_before:
                        hidden_states = layer_module.final_layer_norm(hidden_states)
                    return hidden_states
                return forward

            layer.forward = make_forward(layer, neurons, bias_values)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            transformer_layers[layer_idx].forward = original  # type: ignore[method-assign]

def score_candidate_neurons(
    model: torch.nn.Module,
    windows: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_layers: Iterable[int],
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    layers = sorted({int(l) for l in candidate_layers})
    if _uses_opt_decoder_layers(model, layers):
        return _score_candidate_neurons_opt(model, windows, source_mask, layers, out_dir)
    return _prev_score_candidate_neurons(model, windows, source_mask, layers, out_dir)


def estimate_bias(model: torch.nn.Module, windows: torch.Tensor, selected: torch.Tensor, source_mask: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if selected.numel() and _uses_opt_decoder_layers(model, selected[:, 0].unique().tolist()):
        return _estimate_bias_opt(model, windows, selected, source_mask)
    return _prev_estimate_bias(model, windows, selected, source_mask)


@contextmanager
def patched_bias_intervention(
    model: torch.nn.Module,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
):
    if bias_by_layer and _uses_opt_decoder_layers(model, bias_by_layer.keys()):
        with _patched_opt_bias_intervention(model, bias_by_layer, source_mask_batch, scale, mode):
            yield
        return
    with _prev_patched_bias_intervention(model, bias_by_layer, source_mask_batch, scale, mode):
        yield
