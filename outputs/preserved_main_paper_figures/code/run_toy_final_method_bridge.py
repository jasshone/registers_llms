
from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_realtext_toy_sink_training import RealTextTinyGPT, RealTextToyConfig, fixed_eval_batch, matched_random, split_tokens
from analyze_realtext_toy_checkpoints import attention_qk_loss_chunked

CORE_SCALES = [0.9, 1.0, 1.25, 1.5, 2.0, 3.0]
RESCUE_SCALES = [0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 7.0, 8.0, 10.0]
EXTENDED_SCALES = [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 16.0, 20.0, 24.0, 30.0, 32.0]


def batched(batch: torch.Tensor, chunk_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, int(batch.shape[0]), chunk_size):
        yield batch[start : start + chunk_size]


def checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix('checkpoint_step'))


def load_splits(token_cache: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(token_cache, map_location='cpu')
    if isinstance(payload, dict) and 'tokens' in payload:
        tokens = payload['tokens'].long().cpu()
        meta = payload.get('metadata', {})
    elif torch.is_tensor(payload):
        tokens = payload.long().cpu()
        meta = {}
    else:
        raise ValueError(f'Unsupported token cache: {token_cache}')
    return split_tokens(tokens, train_frac=0.9, val_frac=0.05), meta


def extend_positional_embedding(model: RealTextTinyGPT, new_seq_len: int) -> None:
    old = model.wpe
    if new_seq_len <= old.num_embeddings:
        model.cfg.seq_len = max(model.cfg.seq_len, new_seq_len)
        return
    new = nn.Embedding(new_seq_len, old.embedding_dim).to(old.weight.device)
    with torch.no_grad():
        new.weight[: old.num_embeddings].copy_(old.weight)
        new.weight[old.num_embeddings :].copy_(old.weight[-1:].expand(new_seq_len - old.num_embeddings, -1))
    model.wpe = new
    model.cfg.seq_len = new_seq_len


def positive_episodes(bos_by_layer: torch.Tensor, alpha: float = 0.05) -> list[dict[str, float | int]]:
    slopes = bos_by_layer[1:] - bos_by_layer[:-1]
    if slopes.numel() == 0:
        return []
    max_pos = float(slopes.clamp_min(0).max().item())
    thresh = alpha * max_pos
    return [{'attn_layer': idx, 'prev_mlp_layer': idx - 1, 'slope': float(s.item()), 'threshold': thresh} for idx, s in enumerate(slopes, start=1) if float(s.item()) > thresh and float(s.item()) > 0]


def emergence_layers(bos_by_layer: torch.Tensor, margin: int = 2) -> list[int]:
    if bos_by_layer.numel() == 0:
        return [0]
    first = float(bos_by_layer[0].item())
    mx = float(bos_by_layer.max().item())
    half = first + 0.5 * (mx - first)
    above = torch.nonzero(bos_by_layer >= half, as_tuple=False).flatten()
    half_layer = int(above[0].item()) if above.numel() else int(bos_by_layer.argmax().item())
    slopes = bos_by_layer[1:] - bos_by_layer[:-1]
    max_slope_prev = int(slopes.argmax().item()) if slopes.numel() else 0
    lo = max(0, min(half_layer, max_slope_prev + 1) - margin - 1)
    hi = min(int(bos_by_layer.numel()) - 1, max(half_layer, max_slope_prev + 1) + margin)
    return list(range(lo, hi + 1))


@torch.no_grad()
def clean_bos_by_layer(model: RealTextTinyGPT, batch: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    layer_sum = torch.zeros(model.cfg.n_layer, dtype=torch.float64)
    head_sum = torch.zeros(model.cfg.n_layer, model.cfg.n_head, dtype=torch.float64)
    examples = 0
    loss_sum = 0.0
    token_count = 0
    for chunk in batched(batch, chunk_size):
        out = model(chunk, output_attentions=True)
        y = chunk[:, 1:].contiguous()
        logits = out['logits'][:, :-1, :].float().contiguous()
        loss_sum += float(F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction='sum').cpu())
        token_count += int(y.numel())
        examples += int(chunk.shape[0])
        for layer_idx, attn in enumerate(out['attentions']):
            vals = attn[:, :, 1:, 0].float()
            layer_sum[layer_idx] += vals.mean(dim=(1, 2)).sum().double().cpu()
            head_sum[layer_idx] += vals.mean(dim=2).sum(dim=0).double().cpu()
    by_layer = (layer_sum / max(1, examples)).float()
    by_head = (head_sum / max(1, examples)).float()
    nll = loss_sum / max(1, token_count)
    return by_layer, by_head, nll, math.exp(nll)


@torch.no_grad()
def toy_attention_sink_mask(model: RealTextTinyGPT, batch: torch.Tensor, *, chunk_size: int, threshold: float, min_query_count: int) -> torch.Tensor:
    seq_len = int(batch.shape[1])
    sink_mask = torch.zeros(batch.shape, dtype=torch.bool)
    future_query_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32), diagonal=-1)
    denominators = future_query_mask.sum(dim=0).clamp_min(1.0)
    valid_future = denominators >= int(min_query_count)
    offset = 0
    for chunk in batched(batch, chunk_size):
        out = model(chunk, output_attentions=True)
        batch_hits = torch.zeros((chunk.shape[0], seq_len), dtype=torch.long)
        future = future_query_mask.to(chunk.device)
        denom = denominators.to(chunk.device)
        for attn in out['attentions']:
            received = (attn.float() * future[None, None, :, :]).sum(dim=2) / denom[None, None, :]
            batch_hits += (received.cpu() >= float(threshold)).sum(dim=1)
        candidate = valid_future[None, :].expand(chunk.shape[0], -1).clone()
        sink_mask[offset : offset + chunk.shape[0]] = candidate & (batch_hits > 0)
        offset += int(chunk.shape[0])
    return sink_mask


def make_masks(batch: torch.Tensor, attention_sink_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    bos = torch.zeros(attention_sink_mask.shape, dtype=torch.bool, device=attention_sink_mask.device)
    bos[:, 0] = True
    return {
        'bos_only': bos,
        'pipeline_sinks_plus_bos': attention_sink_mask.bool() | bos,
    }


@torch.no_grad()
def score_candidates(
    model: RealTextTinyGPT,
    batch: torch.Tensor,
    source_mask: torch.Tensor,
    layers: list[int],
    *,
    max_top: int,
    source_abs_bonus: float,
    chunk_size: int,
) -> tuple[torch.Tensor, list[tuple[float, int, int, float, float, float]]]:
    sums_source: dict[int, torch.Tensor] = {}
    sums_ctrl: dict[int, torch.Tensor] = {}
    counts_source: dict[int, int] = {}
    counts_ctrl: dict[int, int] = {}
    width = model.cfg.d_mlp
    for layer in layers:
        sums_source[layer] = torch.zeros(width, dtype=torch.float64)
        sums_ctrl[layer] = torch.zeros(width, dtype=torch.float64)
        counts_source[layer] = 0
        counts_ctrl[layer] = 0
    offset = 0
    for chunk in batched(batch, chunk_size):
        out = model(chunk, capture_mlp=True)
        src = source_mask[offset : offset + chunk.shape[0]].to(chunk.device).bool()
        ctrl = ~src
        ctrl[:, 0] = False
        for layer in layers:
            acts = out['mlp_acts'][layer].float()
            if src.any():
                vals = acts[src].detach().cpu().double()
                sums_source[layer].add_(vals.sum(dim=0))
                counts_source[layer] += int(vals.shape[0])
            if ctrl.any():
                vals = acts[ctrl].detach().cpu().double()
                sums_ctrl[layer].add_(vals.sum(dim=0))
                counts_ctrl[layer] += int(vals.shape[0])
        offset += int(chunk.shape[0])
    rows = []
    for layer in layers:
        source_mean = sums_source[layer] / max(1, counts_source[layer])
        ctrl_mean = sums_ctrl[layer] / max(1, counts_ctrl[layer])
        gap = source_mean - ctrl_mean
        combined = gap + float(source_abs_bonus) * source_mean.abs()
        for neuron, score in enumerate(combined.tolist()):
            rows.append((float(score), layer, neuron, float(gap[neuron].item()), float(source_mean[neuron].item()), float(ctrl_mean[neuron].item())))
    rows.sort(reverse=True)
    selected = torch.tensor([[layer, neuron] for _score, layer, neuron, *_rest in rows[:max_top]], dtype=torch.long) if rows else torch.empty((0, 2), dtype=torch.long)
    return selected, rows[:500]


@torch.no_grad()
def estimate_bias(
    model: RealTextTinyGPT,
    batch: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if selected.numel() == 0:
        return result
    by_layer = [int(x) for x in selected[:, 0].unique(sorted=True).tolist()]
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    for layer in by_layer:
        neurons = selected[selected[:, 0] == layer][:, 1]
        sums[layer] = torch.zeros(neurons.numel(), dtype=torch.float64)
        counts[layer] = 0
    offset = 0
    for chunk in batched(batch, chunk_size):
        out = model(chunk, capture_mlp=True)
        src = source_mask[offset : offset + chunk.shape[0]].to(chunk.device).bool()
        for layer in by_layer:
            neurons = selected[selected[:, 0] == layer][:, 1].to(chunk.device)
            acts = out['mlp_acts'][layer].float()[..., neurons]
            if src.any():
                vals = acts[src].detach().cpu().double()
                sums[layer].add_(vals.sum(dim=0))
                counts[layer] += int(vals.shape[0])
        offset += int(chunk.shape[0])
    for layer in by_layer:
        neurons = selected[selected[:, 0] == layer][:, 1].clone()
        result[layer] = (neurons, (sums[layer] / max(1, counts[layer])).float())
    return result


@contextmanager
def patched_toy_bias_intervention(
    model: RealTextTinyGPT,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_mask_batch: torch.Tensor,
    scale: float,
    mode: str,
) -> Iterator[None]:
    originals: dict[int, Any] = {}
    try:
        for layer, (neurons, bias) in bias_by_layer.items():
            mlp = model.blocks[layer].mlp
            originals[layer] = mlp.forward

            def make_forward(layer_mlp: torch.nn.Module, selected_neurons: torch.Tensor, bias_values: torch.Tensor):
                def forward(
                    x: torch.Tensor,
                    *,
                    zero_bos_neurons: torch.Tensor | None = None,
                    capture: list[torch.Tensor] | None = None,
                ) -> torch.Tensor:
                    h = F.gelu(layer_mlp.c_fc(x))
                    if capture is not None:
                        capture.append(h if h.requires_grad else h.detach())
                    idx = selected_neurons.to(x.device)
                    b = bias_values.to(x.device, dtype=h.dtype) * float(scale)
                    dummy_pos = 1
                    h = h.clone()
                    if mode in {'add_only', 'transfer'}:
                        h[:, dummy_pos, idx] = b
                    if mode in {'subtract_only', 'transfer'}:
                        mask = source_mask_batch.to(x.device).bool()
                        expanded = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
                        expanded[:, 0] = mask[:, 0]
                        expanded[:, 2:] = mask[:, 1:]
                        expanded[:, dummy_pos] = False
                        gate_sel = h[..., idx].clone()
                        if expanded.any():
                            gate_sel[expanded] = gate_sel[expanded] - b
                        h[..., idx] = gate_sel
                    if zero_bos_neurons is not None and zero_bos_neurons.numel() > 0:
                        h[:, 0, zero_bos_neurons.to(h.device)] = 0.0
                    return layer_mlp.c_proj(layer_mlp.dropout(h))

                return forward

            mlp.forward = make_forward(mlp, neurons, bias)  # type: ignore[method-assign]
        yield
    finally:
        for layer, original in originals.items():
            model.blocks[layer].mlp.forward = original  # type: ignore[method-assign]


def build_intervened_embeds(model: RealTextTinyGPT, batch: torch.Tensor) -> torch.Tensor:
    embeds = model.wte(batch)
    dummy = torch.zeros((batch.shape[0], 1, embeds.shape[-1]), dtype=embeds.dtype, device=embeds.device)
    return torch.cat([embeds[:, :1, :], dummy, embeds[:, 1:, :]], dim=1)


def nll_from_shifted(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]).float(), shift_labels.reshape(-1), ignore_index=-100, reduction='none').view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return float((losses * valid).sum().detach().cpu()), int(valid.sum().item())


@torch.no_grad()
def evaluate_with_fixed_bias(
    model: RealTextTinyGPT,
    batch: torch.Tensor,
    selected: torch.Tensor,
    source_mask: torch.Tensor,
    bias_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    mode: str,
    scale: float,
    chunk_size: int,
) -> dict[str, Any]:
    if selected.numel() == 0:
        return {'dummy_minus_bos_mean': float('-inf'), 'ppl_ratio': float('inf'), 'status': 'empty'}
    clean_loss = 0.0
    changed_loss = 0.0
    token_count = 0
    examples = 0
    depth = model.cfg.n_layer
    bos_before = torch.zeros(depth, dtype=torch.float64)
    bos_after = torch.zeros(depth, dtype=torch.float64)
    dummy_after = torch.zeros(depth, dtype=torch.float64)
    offset = 0
    for chunk in batched(batch, chunk_size):
        bsz = int(chunk.shape[0])
        src = source_mask[offset : offset + bsz]
        offset += bsz
        clean = model(chunk, output_attentions=True)
        labels = chunk.clone()
        labels[:, 0] = -100
        loss, count = nll_from_shifted(clean['logits'], labels)
        clean_loss += loss
        token_count += count
        with patched_toy_bias_intervention(model, bias_by_layer, src, scale, mode):
            embeds = build_intervened_embeds(model, chunk)
            changed = model(inputs_embeds=embeds, output_attentions=True)
        ilabels = torch.full((chunk.shape[0], chunk.shape[1] + 1), -100, dtype=torch.long, device=chunk.device)
        ilabels[:, 2:] = chunk[:, 1:]
        iloss, icount = nll_from_shifted(changed['logits'], ilabels)
        if icount != count:
            raise RuntimeError(f'token count mismatch: {icount} != {count}')
        changed_loss += iloss
        examples += bsz
        for layer_idx, attn in enumerate(clean['attentions']):
            bos_before[layer_idx] += attn[:, :, 1:, 0].float().mean(dim=(1, 2)).sum().double().cpu()
        for layer_idx, attn in enumerate(changed['attentions']):
            bos_after[layer_idx] += attn[:, :, 2:, 0].float().mean(dim=(1, 2)).sum().double().cpu()
            dummy_after[layer_idx] += attn[:, :, 2:, 1].float().mean(dim=(1, 2)).sum().double().cpu()
    bos_before = (bos_before / max(1, examples)).float()
    bos_after = (bos_after / max(1, examples)).float()
    dummy_after = (dummy_after / max(1, examples)).float()
    clean_nll = clean_loss / max(1, token_count)
    changed_nll = changed_loss / max(1, token_count)
    ppl_ratio = math.exp(changed_nll - clean_nll)
    return {
        'bos_before_mean': float(bos_before.mean().item()),
        'bos_after_mean': float(bos_after.mean().item()),
        'dummy_after_mean': float(dummy_after.mean().item()),
        'dummy_minus_bos_mean': float((dummy_after - bos_after).mean().item()),
        'baseline_ppl': math.exp(clean_nll),
        'intervened_ppl': math.exp(changed_nll),
        'ppl_ratio': ppl_ratio,
        'ppl_change_pct': (ppl_ratio - 1.0) * 100.0,
        'delta_mean_nll': changed_nll - clean_nll,
        'bos_before_by_layer': ' '.join(f'{float(x):.6g}' for x in bos_before.tolist()),
        'bos_after_by_layer': ' '.join(f'{float(x):.6g}' for x in bos_after.tolist()),
        'dummy_after_by_layer': ' '.join(f'{float(x):.6g}' for x in dummy_after.tolist()),
        'dummy_minus_bos_by_layer': ' '.join(f'{float(x):.6g}' for x in (dummy_after - bos_after).tolist()),
    }


def safe_float(value: Any, default: float = float('nan')) -> float:
    try:
        return float(value)
    except Exception:
        return default


def choose_validation_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transfers = [r for r in rows if r.get('mode') == 'transfer' and r.get('status') == 'ok']
    positive = [r for r in transfers if safe_float(r.get('dummy_minus_bos_mean'), -math.inf) > 0]
    pool = positive or transfers
    strong = [r for r in pool if safe_float(r.get('dummy_minus_bos_mean'), -math.inf) >= 0.30]
    if strong:
        best = min(strong, key=lambda r: (safe_float(r.get('ppl_ratio'), math.inf), int(r['topk']), -safe_float(r.get('dummy_minus_bos_mean'))))
        best['selection_reason'] = 'reloc>=0.30_min_val_ppl'
        return best
    if positive:
        best = min(positive, key=lambda r: (safe_float(r.get('ppl_ratio'), math.inf), int(r['topk']), -safe_float(r.get('dummy_minus_bos_mean'))))
        best['selection_reason'] = 'positive_reloc_min_val_ppl'
        return best
    best = max(transfers, key=lambda r: safe_float(r.get('dummy_minus_bos_mean'), -math.inf))
    best['selection_reason'] = 'no_positive_max_val_reloc'
    return best


def layer_counts(selected: torch.Tensor) -> str:
    if selected.numel() == 0:
        return ''
    parts = []
    for layer in selected[:, 0].unique(sorted=True).tolist():
        parts.append(f'{int(layer)}:{int((selected[:, 0] == int(layer)).sum().item())}')
    return ' '.join(parts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 4) -> str:
    val = safe_float(x)
    if not math.isfinite(val):
        return 'nan'
    return f'{val:.{digits}g}'


def write_summary(path: Path, test_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> None:
    best = max(test_rows, key=lambda r: safe_float(r.get('dummy_minus_bos_mean'), -math.inf)) if test_rows else None
    final = next((r for r in test_rows if int(r['step']) == max(int(x['step']) for x in test_rows)), None) if test_rows else None
    lines = [
        '# Toy Final-Method Bridge',
        '',
        'This reruns the toy checkpoint bridge with the final paper-figure transfer protocol: emergence-band candidate layers, source-control MLP scoring with source_abs_bonus=0, validation selection over mask/top-k/scale, and fixed source-bias transfer to a zero dummy slot.',
        '',
        '| step | mask | k | scale | val reloc | val PPL | test reloc | test PPL | BOS before | dummy after | reason |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in test_rows:
        lines.append(
            f"| {int(r['step'])} | {r['mask']} | {int(r['topk'])} | {fmt(r['scale'])} | {fmt(r['val_dummy_minus_bos_mean'])} | {fmt(r['val_ppl_ratio'])} | "
            f"{fmt(r['dummy_minus_bos_mean'])} | {fmt(r['ppl_ratio'])} | {fmt(r['bos_before_mean'])} | {fmt(r['dummy_after_mean'])} | {r.get('selection_reason','')} |"
        )
    if best:
        lines += ['', f"Strongest test relocation: step `{int(best['step'])}` with `{fmt(best['dummy_minus_bos_mean'])}` dummy-minus-BOS, PPL ratio `{fmt(best['ppl_ratio'])}`, mask `{best['mask']}`, k `{int(best['topk'])}`, scale `{fmt(best['scale'])}`."]
    if final:
        lines.append(f"Final checkpoint: dummy-minus-BOS `{fmt(final['dummy_minus_bos_mean'])}`, PPL ratio `{fmt(final['ppl_ratio'])}`, BOS-before mean `{fmt(final['bos_before_mean'])}`, dummy-after mean `{fmt(final['dummy_after_mean'])}`.")
    positives = sum(safe_float(r.get('dummy_minus_bos_mean'), -math.inf) > 0 for r in test_rows)
    lines.append(f"Positive test relocation appears in `{positives}/{len(test_rows)}` checkpoints under validation-selected rows.")
    path.write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run final paper-figure fixed-bias transfer method on real-text toy checkpoints.')
    p.add_argument('--run-dir', type=Path, default=Path('outputs/realtext_toy_sink_training/wikitext_pythia_tokenizer_10k_eval128'))
    p.add_argument('--token-cache', type=Path, default=Path('outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt'))
    p.add_argument('--out', type=Path, default=Path('outputs/toy_pretrained_bridge/final_method_wikitext_pythia_tokenizer'))
    p.add_argument('--steps', nargs='*', type=int, default=None)
    p.add_argument('--selection-examples', type=int, default=512)
    p.add_argument('--bias-examples', type=int, default=512)
    p.add_argument('--eval-examples', type=int, default=512)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--topks', nargs='*', type=int, default=[1, 2, 4, 8, 16])
    p.add_argument('--scale-preset', choices=['core', 'rescue', 'extended'], default='rescue')
    p.add_argument('--scales', nargs='*', type=float, default=None)
    p.add_argument('--source-abs-bonus', type=float, default=0.0)
    p.add_argument('--attention-threshold', type=float, default=0.2)
    p.add_argument('--min-query-count', type=int, default=32)
    p.add_argument('--masks', nargs='*', default=['bos_only', 'pipeline_sinks_plus_bos'])
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return p.parse_args()


def scale_values(args: argparse.Namespace) -> list[float]:
    if args.scales:
        return sorted(set(float(x) for x in args.scales))
    if args.scale_preset == 'core':
        return CORE_SCALES
    if args.scale_preset == 'rescue':
        return RESCUE_SCALES
    return EXTENDED_SCALES


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'cpu' else 'cuda'))
    splits, token_meta = load_splits(args.token_cache)
    ckpts = sorted(args.run_dir.glob('checkpoint_step*.pt'), key=checkpoint_step)
    if args.steps is not None:
        keep = set(args.steps)
        ckpts = [p for p in ckpts if checkpoint_step(p) in keep]
    if not ckpts:
        raise ValueError(f'no checkpoints found in {args.run_dir}')
    args.out.mkdir(parents=True, exist_ok=True)
    scales = scale_values(args)
    all_val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for ckpt_path in ckpts:
        payload = torch.load(ckpt_path, map_location='cpu')
        cfg = RealTextToyConfig(**payload['config'])
        model = RealTextTinyGPT(cfg).to(device)
        model.load_state_dict(payload['model'])
        model.eval()
        clean_seq_len = int(cfg.seq_len)
        train_batch = fixed_eval_batch(splits['train'], cfg, args.selection_examples, device)
        bias_batch = fixed_eval_batch(splits['train'], cfg, args.bias_examples, device)
        val_batch = fixed_eval_batch(splits['val'], cfg, args.eval_examples, device)
        test_batch = fixed_eval_batch(splits['test'], cfg, args.eval_examples, device)
        extend_positional_embedding(model, clean_seq_len + 1)
        step = int(payload['step'])
        train_bos_layer, train_bos_head, train_nll, train_ppl = clean_bos_by_layer(model, train_batch, args.chunk_size)
        candidate_layers = emergence_layers(train_bos_layer)
        need_sink_mask = 'pipeline_sinks_plus_bos' in set(args.masks)
        if need_sink_mask:
            train_sink = toy_attention_sink_mask(model, train_batch, chunk_size=args.chunk_size, threshold=args.attention_threshold, min_query_count=args.min_query_count)
            bias_sink = toy_attention_sink_mask(model, bias_batch, chunk_size=args.chunk_size, threshold=args.attention_threshold, min_query_count=args.min_query_count)
            val_sink = toy_attention_sink_mask(model, val_batch, chunk_size=args.chunk_size, threshold=args.attention_threshold, min_query_count=args.min_query_count)
            test_sink = toy_attention_sink_mask(model, test_batch, chunk_size=args.chunk_size, threshold=args.attention_threshold, min_query_count=args.min_query_count)
        else:
            train_sink = torch.zeros(train_batch.shape, dtype=torch.bool)
            bias_sink = torch.zeros(bias_batch.shape, dtype=torch.bool)
            val_sink = torch.zeros(val_batch.shape, dtype=torch.bool)
            test_sink = torch.zeros(test_batch.shape, dtype=torch.bool)
        masks_by_split = {
            'train': make_masks(train_batch, train_sink),
            'bias': make_masks(bias_batch, bias_sink),
            'val': make_masks(val_batch, val_sink),
            'test': make_masks(test_batch, test_sink),
        }
        context = {
            'step': step,
            'candidate_layers': candidate_layers,
            'train_bos_by_layer': ' '.join(f'{float(x):.6g}' for x in train_bos_layer.tolist()),
            'train_max_head_bos': float(train_bos_head.max().item()),
            'train_mean_bos': float(train_bos_layer.mean().item()),
            'train_ppl': train_ppl,
            'train_episodes': positive_episodes(train_bos_layer),
            'attention_sink_positions_train': int(train_sink.sum().item()),
            'attention_sink_windows_train': int(train_sink.any(dim=1).sum().item()),
        }
        (args.out / f'context_step{step}.json').write_text(json.dumps(context, indent=2, default=str) + '\n')
        val_rows: list[dict[str, Any]] = []
        selected_cache: dict[tuple[str, int], tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]], str]] = {}
        for mask_name in args.masks:
            candidates, score_rows = score_candidates(model, train_batch, masks_by_split['train'][mask_name], candidate_layers, max_top=max(args.topks), source_abs_bonus=args.source_abs_bonus, chunk_size=args.chunk_size)
            write_csv(args.out / f'score_rows_step{step}_{mask_name}.csv', [
                {'score': a, 'layer': b, 'neuron': c, 'gap': d, 'source_mean': e, 'control_mean': f}
                for a, b, c, d, e, f in score_rows
            ])
            for topk in args.topks:
                if topk > int(candidates.shape[0]):
                    continue
                selected = candidates[:topk].clone()
                bias = estimate_bias(model, bias_batch, selected, masks_by_split['bias'][mask_name], chunk_size=args.chunk_size)
                selected_cache[(mask_name, topk)] = (selected, bias, layer_counts(selected))
                specs = [('add_only', 1.0), ('subtract_only', 1.0)] + [('transfer', s) for s in scales]
                for mode, scale in specs:
                    row: dict[str, Any] = {
                        'step': step,
                        'split': 'val',
                        'mask': mask_name,
                        'mode': mode,
                        'topk': topk,
                        'scale': float(scale),
                        'layers': layer_counts(selected),
                        'candidate_layers': ' '.join(str(x) for x in candidate_layers),
                        'source_abs_bonus': float(args.source_abs_bonus),
                        'sink_positions': int(masks_by_split['val'][mask_name].sum().item()),
                        'status': 'ok',
                    }
                    try:
                        row.update(evaluate_with_fixed_bias(model, val_batch, selected.to(device), masks_by_split['val'][mask_name], bias, mode=mode, scale=float(scale), chunk_size=args.chunk_size))
                    except Exception as exc:
                        row['status'] = 'error'
                        row['error'] = repr(exc)
                    val_rows.append(row)
                    all_val_rows.append(row)
                    print(f"[val] step={step} {mask_name} k={topk} {mode} s={float(scale):g}: reloc={fmt(row.get('dummy_minus_bos_mean'))} ppl={fmt(row.get('ppl_ratio'))} status={row['status']}", flush=True)
                    write_csv(args.out / 'validation_grid.csv', all_val_rows)
        best = choose_validation_row(val_rows)
        mask_name = str(best['mask'])
        topk = int(best['topk'])
        selected, bias, layers = selected_cache[(mask_name, topk)]
        test_row: dict[str, Any] = {
            'step': step,
            'split': 'test',
            'mask': mask_name,
            'mode': 'transfer',
            'topk': topk,
            'scale': float(best['scale']),
            'layers': layers,
            'candidate_layers': ' '.join(str(x) for x in candidate_layers),
            'source_abs_bonus': float(args.source_abs_bonus),
            'selection_reason': best.get('selection_reason', ''),
            'val_dummy_minus_bos_mean': best.get('dummy_minus_bos_mean'),
            'val_ppl_ratio': best.get('ppl_ratio'),
            'sink_positions': int(masks_by_split['test'][mask_name].sum().item()),
            'status': 'ok',
        }
        test_row.update(evaluate_with_fixed_bias(model, test_batch, selected.to(device), masks_by_split['test'][mask_name], bias, mode='transfer', scale=float(best['scale']), chunk_size=args.chunk_size))
        test_rows.append(test_row)
        write_csv(args.out / 'test_selected_rows.csv', test_rows)
        write_summary(args.out / 'toy_final_method_bridge_summary.md', test_rows, all_val_rows)
        print(f"[test] step={step}: mask={mask_name} k={topk} s={float(best['scale']):g} reloc={test_row['dummy_minus_bos_mean']:.4g} ppl={test_row['ppl_ratio']:.4g}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.out / 'config.json').write_text(json.dumps({'args': vars(args), 'token_cache_metadata': token_meta, 'scales': scales}, indent=2, default=str) + '\n')
    print(args.out / 'toy_final_method_bridge_summary.md')


if __name__ == '__main__':
    main()
