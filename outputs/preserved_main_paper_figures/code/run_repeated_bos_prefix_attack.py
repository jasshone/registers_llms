
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from sink_neurons.artifacts import load_pt
from sink_neurons.env import configure_runtime
from sink_neurons.intervention import _split_mlp_forward, make_layer_assignment_map
from sink_neurons.modeling import get_embed_tokens, get_mlp_intermediate_size, get_transformer_layers, load_model, resolve_device

DEFAULT_MODEL_ID = 'EleutherAI/pythia-1b'
DEFAULT_SELECTION = Path('outputs/attention_jump_ablation/pythia_1b/selections/full_percentile.pt')
DEFAULT_WINDOWS = Path('outputs/final_split_bias_transfer_source_abs_bonus0_official_scope/pythia_1b/windows_192x1024.pt')
DEFAULT_OUT = Path('outputs/sink_hijacking_attack/repeated_bos_prefix/pythia_1b')




def load_selected(path: Path) -> torch.Tensor:
    payload = load_pt(path)
    if isinstance(payload.get('tensors'), dict) and 'selected' in payload['tensors']:
        return payload['tensors']['selected'].long()
    if 'selected' in payload:
        return payload['selected'].long()
    if 'selected_top' in payload:
        return payload['selected_top'].long()
    raise KeyError(path)


def load_windows(path: Path, n: int) -> torch.Tensor:
    payload = load_pt(path)
    if isinstance(payload.get('tensors'), dict) and 'windows' in payload['tensors']:
        windows = payload['tensors']['windows']
    elif 'windows' in payload:
        windows = payload['windows']
    else:
        raise KeyError(path)
    return windows[:n].long()


def layer_matched_random_selection(model, selected: torch.Tensor, *, seed: int) -> torch.Tensor:
    width = get_mlp_intermediate_size(model)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    rows: list[tuple[int, int]] = []
    by_layer: dict[int, set[int]] = {}
    for layer_idx, neuron_idx in selected.cpu().tolist():
        by_layer.setdefault(int(layer_idx), set()).add(int(neuron_idx))
    for layer_idx in sorted(by_layer):
        excluded = by_layer[layer_idx]
        count = len(excluded)
        pool = torch.tensor([idx for idx in range(width) if idx not in excluded], dtype=torch.long)
        if pool.numel() < count:
            raise ValueError(f'layer {layer_idx} has only {pool.numel()} random-control neurons for {count} selected')
        chosen = pool[torch.randperm(pool.numel(), generator=generator)[:count]]
        rows.extend((layer_idx, int(neuron_idx)) for neuron_idx in chosen.tolist())
    return torch.tensor(rows, dtype=torch.long)


@contextmanager
def patched_mlp_reroute_at_start(model, selected: torch.Tensor, *, dummy_start: int, num_dummy_tokens: int, relocation_scale: float, relocation_fraction: float = 1.0, preserve_original: bool = False) -> Iterator[None]:
    originals: dict[int, Any] = {}
    try:
        assignment_map = make_layer_assignment_map(selected, num_dummy_tokens=num_dummy_tokens, assignment_strategy='round_robin')
        for layer_idx, (neurons, assignments) in assignment_map.items():
            layer = get_transformer_layers(model)[layer_idx]
            mlp = layer.mlp
            originals[layer_idx] = mlp.forward

            def make_forward(layer_mlp, selected_neurons: torch.Tensor, dummy_assignments: torch.Tensor):
                def forward(x: torch.Tensor) -> torch.Tensor:
                    selected_on_device = selected_neurons.to(x.device)
                    gate_act, project = _split_mlp_forward(layer_mlp, x)
                    gate_selected = gate_act[..., selected_on_device].clone()
                    start = max(0, min(int(dummy_start), x.shape[1] - num_dummy_tokens))
                    original_positions = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                    original_positions[start:start + num_dummy_tokens] = False
                    original_selected = gate_selected[:, original_positions, :]
                    max_vals = original_selected.amax(dim=1)
                    if not preserve_original:
                        gate_selected[:, original_positions, :] = original_selected * (1.0 - relocation_fraction)
                    assignments_on_device = dummy_assignments.to(x.device)
                    for dummy_idx in range(num_dummy_tokens):
                        assigned = assignments_on_device == dummy_idx
                        gate_selected[:, start + dummy_idx, assigned] = max_vals[:, assigned] * relocation_scale * relocation_fraction
                    gate_act = gate_act.clone()
                    gate_act[..., selected_on_device] = gate_selected
                    return project(gate_act)
                return forward

            mlp.forward = make_forward(mlp, neurons, assignments)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            get_transformer_layers(model)[layer_idx].mlp.forward = original  # type: ignore[method-assign]

@dataclass(frozen=True)
class Config:
    model_id: str
    selection: str
    windows: str
    num_windows: int
    batch_size: int
    counts: list[int]
    scale: float
    random_controls: int
    seed: int


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


def parse_counts(text: str) -> list[int]:
    return [int(x) for x in text.split(',') if x]


def dtype_from_name(name: str) -> torch.dtype:
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[name]


def build_prefix_inputs(model, input_ids: torch.Tensor, *, count: int, init: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeds = get_embed_tokens(model)(input_ids)
    if count <= 0:
        labels = input_ids.clone()
        labels[:, 0] = -100
        mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        return embeds, mask, labels
    if init == 'bos':
        block = embeds[:, :1, :].expand(-1, count, -1).clone()
    elif init == 'ordinary_token':
        src = 1 if input_ids.shape[1] > 1 else 0
        block = embeds[:, src:src + 1, :].expand(-1, count, -1).clone()
    elif init == 'zero':
        block = torch.zeros((input_ids.shape[0], count, embeds.shape[-1]), dtype=embeds.dtype, device=embeds.device)
    else:
        raise ValueError(init)
    # Insert immediately after the original first token/BOS, so original key 0 stays globally visible.
    new_embeds = torch.cat([embeds[:, :1, :], block, embeds[:, 1:, :]], dim=1)
    attention_mask = torch.ones(new_embeds.shape[:2], dtype=torch.long, device=input_ids.device)
    labels = torch.full((input_ids.shape[0], input_ids.shape[1] + count), -100, dtype=torch.long, device=input_ids.device)
    labels[:, 1 + count:] = input_ids[:, 1:]
    return new_embeds, attention_mask, labels


def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction='none').view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return (losses * valid).sum(dim=1), valid.sum(dim=1)


def attention_metrics(attentions: tuple[torch.Tensor, ...], *, count: int) -> dict[str, float]:
    query_start = min(1 + count, attentions[0].shape[-2] - 1)
    bos_layers = []
    block_layers = []
    for attn in attentions:
        a = attn.float()
        bos_layers.append(a[:, :, query_start:, 0].mean().detach().cpu())
        if count > 0:
            block_layers.append(a[:, :, query_start:, 1:1 + count].sum(dim=-1).mean().detach().cpu())
        else:
            block_layers.append(torch.tensor(float('nan')))
    bos = torch.stack(bos_layers)
    block = torch.stack(block_layers)
    return {
        'bos_attention_mean': float(bos.mean()),
        'bos_attention_max_layer': float(bos.max()),
        'inserted_attention_mean': float(block.mean()),
        'inserted_attention_max_layer': float(block.max()),
        'inserted_minus_bos_mean': float((block - bos).mean()) if count > 0 else float('nan'),
    }


@torch.no_grad()
def evaluate(model, windows: torch.Tensor, *, condition: str, count: int, batch_size: int, selected: torch.Tensor | None, scale: float, seed: int) -> dict[str, Any]:
    device = resolve_device(model)
    loss_sum = 0.0
    token_sum = 0
    metric_sums: dict[str, float] = {}
    examples = 0
    if condition == 'random_copy_bos':
        if selected is None:
            raise ValueError('selected required for random_copy_bos')
        selected_for_condition = layer_matched_random_selection(model, selected, seed=seed)
    else:
        selected_for_condition = selected
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start:start + batch_size].to(device)
        if condition == 'clean':
            outputs = model(input_ids=batch, output_attentions=True, use_cache=False, return_dict=True)
            labels = batch.clone(); labels[:, 0] = -100
            metrics = attention_metrics(outputs.attentions, count=0)
        else:
            init = 'ordinary_token' if condition == 'ordinary_repeat_only' else 'bos'
            embeds, mask, labels = build_prefix_inputs(model, batch, count=count, init=init)
            ctx = nullcontext()
            if condition in {'selected_copy_bos', 'random_copy_bos'}:
                if selected_for_condition is None:
                    raise ValueError(condition)
                ctx = patched_mlp_reroute_at_start(model, selected_for_condition, dummy_start=1, num_dummy_tokens=count, relocation_scale=scale, preserve_original=False)
            elif condition not in {'bos_repeat_only', 'ordinary_repeat_only'}:
                raise ValueError(condition)
            with ctx:
                outputs = model(inputs_embeds=embeds, attention_mask=mask, output_attentions=True, use_cache=False, return_dict=True)
            metrics = attention_metrics(outputs.attentions, count=count)
        loss, tok = nll_from_logits(outputs.logits, labels)
        bsz = int(batch.shape[0])
        loss_sum += float(loss.sum().detach().cpu())
        token_sum += int(tok.sum().detach().cpu())
        examples += bsz
        for key, val in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(val) * bsz
        del outputs
    mean_nll = loss_sum / max(token_sum, 1)
    out = {'mean_nll': mean_nll, 'ppl': math.exp(mean_nll), 'token_count': token_sum, 'num_windows': int(windows.shape[0])}
    out.update({k: v / max(examples, 1) for k, v in metric_sums.items()})
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float('nan')
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / (dx * dy) if dx > 0 and dy > 0 else float('nan')


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    clean = next(r for r in rows if r['condition'] == 'clean')
    nonclean = [r for r in rows if r['condition'] != 'clean']
    attn_ppl_r = pearson([float(r['inserted_attention_mean']) for r in nonclean], [float(r['ppl_ratio_vs_clean']) for r in nonclean])
    attn_delta_ppl_r = pearson([float(r['inserted_minus_bos_mean']) for r in nonclean], [float(r['ppl_ratio_vs_clean']) for r in nonclean])
    selected = [r for r in nonclean if r['condition'] == 'selected_copy_bos']
    strongest = max(selected, key=lambda r: float(r['ppl_ratio_vs_clean'])) if selected else None
    lines = [
        '# Repeated-BOS Prefix Attack',
        '',
        f"Clean PPL: {clean['ppl']:.4g}; clean BOS attention mean: {clean['bos_attention_mean']:.4f}.",
        '',
        'Mechanism checks:',
        '',
        f"- Pearson r(inserted attention, PPL ratio) across non-clean rows: {attn_ppl_r:.4f}.",
        f"- Pearson r(inserted-minus-BOS attention, PPL ratio) across non-clean rows: {attn_delta_ppl_r:.4f}.",
    ]
    if strongest is not None:
        count = int(strongest['count'])
        same_count = {r['condition']: r for r in nonclean if int(r['count']) == count}
        random_row = same_count.get('random_copy_bos')
        ordinary_row = same_count.get('ordinary_repeat_only')
        if random_row is not None:
            lines.append(f"- Strongest selected-copy count {count}: PPL ratio {strongest['ppl_ratio_vs_clean']:.4f} vs layer-matched random {random_row['ppl_ratio_vs_clean']:.4f}; inserted attention {strongest['inserted_attention_mean']:.4f} vs {random_row['inserted_attention_mean']:.4f}.")
        if ordinary_row is not None:
            lines.append(f"- Same count ordinary-token repeat: PPL ratio {ordinary_row['ppl_ratio_vs_clean']:.4f}; inserted attention {ordinary_row['inserted_attention_mean']:.4f}.")
    lines.extend([
        '',
        '| count | condition | inserted attn | BOS attn | inserted-BOS | PPL ratio |',
        '|---:|---|---:|---:|---:|---:|',
    ])
    for r in rows:
        if r['condition'] == 'clean':
            continue
        lines.append(f"| {r['count']} | {r['condition']} | {r['inserted_attention_mean']:.4f} | {r['bos_attention_mean']:.4f} | {r['inserted_minus_bos_mean']:+.4f} | {r['ppl_ratio_vs_clean']:.4f} |")
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    configure_runtime()
    ap = argparse.ArgumentParser(description='Repeated BOS-prefix sink attack/control sweep.')
    ap.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    ap.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    ap.add_argument('--windows', type=Path, default=DEFAULT_WINDOWS)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--num-windows', type=int, default=192)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--counts', default='1,2,4,8,16,32,64')
    ap.add_argument('--scale', type=float, default=4.0)
    ap.add_argument('--random-controls', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    counts = parse_counts(args.counts)
    selected = load_selected(args.selection)
    windows = load_windows(args.windows, args.num_windows)
    model = load_model(args.model_id, dtype=dtype_from_name(args.dtype), eager_attention=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Config(args.model_id, str(args.selection), str(args.windows), args.num_windows, args.batch_size, counts, args.scale, args.random_controls, args.seed)
    (args.out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    rows: list[dict[str, Any]] = []
    clean = evaluate(model, windows, condition='clean', count=0, batch_size=args.batch_size, selected=None, scale=0.0, seed=args.seed)
    rows.append({'condition': 'clean', 'count': 0, 'control_id': -1, 'scale': 0.0, **clean, 'ppl_ratio_vs_clean': 1.0})
    for count in counts:
        for condition in ['bos_repeat_only', 'ordinary_repeat_only', 'selected_copy_bos']:
            res = evaluate(model, windows, condition=condition, count=count, batch_size=args.batch_size, selected=selected, scale=args.scale, seed=args.seed)
            rows.append({'condition': condition, 'count': count, 'control_id': -1, 'scale': args.scale if condition == 'selected_copy_bos' else 0.0, **res, 'ppl_ratio_vs_clean': res['ppl'] / clean['ppl']})
        for ridx in range(args.random_controls):
            res = evaluate(model, windows, condition='random_copy_bos', count=count, batch_size=args.batch_size, selected=selected, scale=args.scale, seed=args.seed * 1000 + count * 10 + ridx)
            rows.append({'condition': 'random_copy_bos', 'count': count, 'control_id': ridx, 'scale': args.scale, **res, 'ppl_ratio_vs_clean': res['ppl'] / clean['ppl']})
        write_csv(args.out / 'repeated_bos_prefix_metrics.csv', rows)
        write_summary(args.out / 'repeated_bos_prefix_summary.md', rows)
    print(args.out / 'repeated_bos_prefix_summary.md')


if __name__ == '__main__':
    main()
