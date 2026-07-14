
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from run_realtext_toy_bos_presence_controls import make_variant_batch, parse_variant, write_csv, mean_attention_to_key
from run_realtext_toy_sink_training import RealTextTinyGPT, RealTextToyConfig, split_tokens


def load_token_cache(path: Path, max_tokens: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location='cpu')
    tokens = payload['tokens'].long()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens, dict(payload.get('metadata', {}))


def qk_pressure_to_key(qk_logits: list[torch.Tensor], key_pos: int) -> tuple[float, float, str]:
    vals = []
    by_layer = []
    for logits in qk_logits:
        grad = logits.grad
        if grad is None:
            by_layer.append(float('nan'))
            continue
        q_start = key_pos + 1
        if q_start >= grad.shape[-2]:
            by_layer.append(float('nan'))
            continue
        pressure_by_head = -grad[:, :, q_start:, key_pos].float().mean(dim=(0, 2))
        vals.append(pressure_by_head.detach().cpu())
        by_layer.append(float(pressure_by_head.mean().detach().cpu()))
    if not vals:
        return float('nan'), float('nan'), ''
    heads = torch.stack(vals)
    finite = heads[torch.isfinite(heads)]
    return float(finite.mean()), float(finite.max()), ' '.join(f'{x:.6g}' for x in by_layer)


def gradient_metrics(model: RealTextTinyGPT, batch: torch.Tensor, bos_positions: list[int], first_n: int) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    out = model(batch, labels=batch, output_attentions=True, output_qk_logits=True)
    for logits in out['qk_logits']:
        logits.retain_grad()
    loss = out['loss']
    assert loss is not None
    loss.backward()

    row: dict[str, Any] = {'grad_loss': float(loss.detach().cpu())}
    pos0_mean, pos0_max, pos0_layers = qk_pressure_to_key(out['qk_logits'], 0)
    row.update({
        'pos0_qk_pressure_mean': pos0_mean,
        'pos0_qk_pressure_max_head': pos0_max,
        'pos0_qk_pressure_by_layer': pos0_layers,
    })
    key_pressures = []
    limit = min(first_n, batch.shape[1] - 1)
    for key_pos in range(limit):
        m, mx, layers = qk_pressure_to_key(out['qk_logits'], key_pos)
        row[f'keypos{key_pos}_qk_pressure_mean'] = m
        row[f'keypos{key_pos}_qk_pressure_max_head'] = mx
        row[f'keypos{key_pos}_qk_pressure_by_layer'] = layers
        if key_pos > 0 and math.isfinite(m):
            key_pressures.append(m)
    row['nonpos0_firstn_qk_pressure_mean'] = float(statistics.fmean(key_pressures)) if key_pressures else float('nan')
    row['nonpos0_firstn_qk_pressure_max'] = float(max(key_pressures)) if key_pressures else float('nan')
    row['pos0_minus_nonpos0_firstn_qk_pressure'] = row['pos0_qk_pressure_mean'] - row['nonpos0_firstn_qk_pressure_mean'] if math.isfinite(row['nonpos0_firstn_qk_pressure_mean']) else float('nan')

    bos_pressures = []
    for pos in bos_positions:
        m, mx, layers = qk_pressure_to_key(out['qk_logits'], pos)
        row[f'bos_pos{pos}_qk_pressure_mean'] = m
        row[f'bos_pos{pos}_qk_pressure_max_head'] = mx
        row[f'bos_pos{pos}_qk_pressure_by_layer'] = layers
        bos_pressures.append(m)
    row['bos_qk_pressure_sum_mean'] = float(sum(p for p in bos_pressures if math.isfinite(p))) if bos_pressures else 0.0
    row['bos_qk_pressure_max_position_mean'] = float(max([p for p in bos_pressures if math.isfinite(p)], default=float('nan'))) if bos_pressures else float('nan')

    row['eval_loss'] = float(loss.detach().cpu())
    # Reuse attentions from the gradient forward. A second no-grad attention
    # forward would double probe cost and is unnecessary for these metrics.
    a0, a0max, a0layers = mean_attention_to_key(out['attentions'], 0)
    row['pos0_attention_mean'] = a0
    row['pos0_attention_max_head'] = a0max
    bos_attns = []
    for pos in bos_positions:
        m, mx, layers = mean_attention_to_key(out['attentions'], pos)
        row[f'bos_pos{pos}_attention_mean'] = m
        row[f'bos_pos{pos}_attention_max_head'] = mx
        bos_attns.append(m)
    row['bos_attention_sum_mean'] = float(sum(bos_attns)) if bos_attns else 0.0
    return row



def parse_step_set(text: str) -> set[int] | None:
    text = (text or '').strip()
    if not text:
        return None
    return {int(x) for x in text.split(',') if x.strip()}

def train_one(args: argparse.Namespace, variant: str, bos_positions: list[int], seed: int, splits: dict[str, torch.Tensor], cfg: RealTextToyConfig, device: torch.device) -> list[dict[str, Any]]:
    torch.manual_seed(seed + 10007 * len(bos_positions) + sum((i + 1) * (p + 31) for i, p in enumerate(bos_positions)))
    random.seed(seed)
    model = RealTextTinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    probe_batch = make_variant_batch(splits[args.eval_split], cfg, args.probe_batch_size, device, bos_positions=bos_positions, fixed=True)
    rows = []
    out_dir = args.out / f'seed_{seed}' / variant.replace(',', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    gradient_steps = parse_step_set(args.gradient_steps)
    for step in range(args.steps + 1):
        should_probe = step % args.eval_interval == 0 and (gradient_steps is None or step in gradient_steps or step == args.steps)
        if should_probe:
            model.eval()
            row = {
                'variant': variant,
                'bos_positions': ' '.join(str(p) for p in bos_positions) if bos_positions else 'none',
                'seed': seed,
                'step': step,
            }
            row.update(gradient_metrics(model, probe_batch, bos_positions, args.first_n_positions))
            rows.append(row)
            write_csv(out_dir / 'gradient_metrics.csv', rows)
            print(f"[{variant} seed={seed}] step={step} pos0_grad={row['pos0_qk_pressure_mean']:+.3e} bos_grad={row['bos_qk_pressure_max_position_mean']:+.3e} pos0_attn={row['pos0_attention_mean']:.4f} bos_attn={row['bos_attention_sum_mean']:.4f}", flush=True)
            model.train()
        if step == args.steps:
            break
        batch = make_variant_batch(splits['train'], cfg, args.batch_size, device, bos_positions=bos_positions, fixed=False)
        out = model(batch, labels=batch)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return rows


def aggregate_final(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for r in rows:
        if r['variant'] not in variants:
            variants.append(r['variant'])
    metrics = ['eval_loss','pos0_attention_mean','bos_attention_sum_mean','pos0_qk_pressure_mean','nonpos0_firstn_qk_pressure_mean','pos0_minus_nonpos0_firstn_qk_pressure','bos_qk_pressure_max_position_mean','bos_qk_pressure_sum_mean']
    out = []
    for v in variants:
        group = [r for r in rows if r['variant'] == v]
        item: dict[str, Any] = {'variant': v, 'bos_positions': group[0]['bos_positions'], 'n_seeds': len(group), 'seeds': ' '.join(str(r['seed']) for r in group)}
        for m in metrics:
            vals = [float(r[m]) for r in group if m in r and math.isfinite(float(r[m]))]
            item[f'{m}_mean'] = float(statistics.fmean(vals)) if vals else float('nan')
            item[f'{m}_std'] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        '# Shifted-BOS Gradient Pressure on Wikitext Train Text',
        '',
        '`pressure = -dL/d attention_logit`; positive means locally increasing attention to that key would reduce loss.',
        '',
        '| variant | BOS positions | seeds | pos0 attn | BOS attn sum | pos0 QK pressure | BOS QK pressure | non-pos0 firstN pressure | pos0 - nonpos0 pressure |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['bos_positions']} | {r['n_seeds']} | "
            f"{r['pos0_attention_mean_mean']:.4f} +/- {r['pos0_attention_mean_std']:.4f} | "
            f"{r['bos_attention_sum_mean_mean']:.4f} +/- {r['bos_attention_sum_mean_std']:.4f} | "
            f"{r['pos0_qk_pressure_mean_mean']:+.3e} +/- {r['pos0_qk_pressure_mean_std']:.1e} | "
            f"{r['bos_qk_pressure_max_position_mean_mean']:+.3e} +/- {r['bos_qk_pressure_max_position_mean_std']:.1e} | "
            f"{r['nonpos0_firstn_qk_pressure_mean_mean']:+.3e} +/- {r['nonpos0_firstn_qk_pressure_mean_std']:.1e} | "
            f"{r['pos0_minus_nonpos0_firstn_qk_pressure_mean']:+.3e} +/- {r['pos0_minus_nonpos0_firstn_qk_pressure_std']:.1e} |"
        )
    path.write_text('\n'.join(lines) + '\n')



def read_existing_final(path: Path, expected_step: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    last = rows[-1]
    try:
        if int(float(last.get('step', -1))) != expected_step:
            return None
    except Exception:
        return None
    out: dict[str, Any] = {}
    for key, value in last.items():
        if value is None:
            out[key] = value
            continue
        text = str(value)
        if key in {'variant', 'bos_positions'}:
            out[key] = text
        elif key in {'seed', 'step'}:
            out[key] = int(float(text))
        else:
            try:
                out[key] = float(text)
            except ValueError:
                out[key] = text
    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Measure QK gradient pressure for BOS-at-0 vs shifted/no-BOS toy controls.')
    ap.add_argument('--token-cache', type=Path, default=Path('outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt'))
    ap.add_argument('--out', type=Path, default=Path('outputs/realtext_toy_shifted_bos_gradients/wikitext103_train'))
    ap.add_argument('--variants', nargs='*', default=['single_0','no_bos','single_8','single_32'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[0,1,2])
    ap.add_argument('--max-tokens', type=int, default=None)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--probe-batch-size', type=int, default=64)
    ap.add_argument('--eval-interval', type=int, default=1000)
    ap.add_argument('--gradient-steps', default='', help='Comma-separated training steps for expensive gradient probes. Empty means every eval interval. Final step is always probed.')
    ap.add_argument('--eval-split', choices=['val','test'], default='val')
    ap.add_argument('--first-n-positions', type=int, default=16)
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--n-layer', type=int, default=4)
    ap.add_argument('--n-head', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--d-mlp', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    ap.add_argument('--no-resume', action='store_true', help='Do not skip seed/variant runs whose gradient_metrics.csv already reached --steps.')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device if args.device != 'auto' else 'cpu')
    tokens, metadata = load_token_cache(args.token_cache, args.max_tokens)
    splits = split_tokens(tokens, train_frac=0.9, val_frac=0.05)
    cfg = RealTextToyConfig(vocab_size=int(metadata.get('vocab_size', 50277)), bos_token_id=int(metadata.get('bos_token_id', 0)), seq_len=args.seq_len, n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model, d_mlp=args.d_mlp)
    variants = [parse_variant(v) for v in args.variants]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps({'args': vars(args), 'model': asdict(cfg), 'metadata': metadata, 'split_tokens': {k: int(v.numel()) for k,v in splits.items()}, 'variants': [{'name': n, 'bos_positions': ps} for n, ps in variants]}, indent=2, default=str) + '\n')
    final_rows = []
    for seed in args.seeds:
        for name, positions in variants:
            metrics_path = args.out / f'seed_{seed}' / name.replace(',', '_') / 'gradient_metrics.csv'
            existing = None if args.no_resume else read_existing_final(metrics_path, args.steps)
            if existing is not None:
                print(f'[resume] seed={seed} variant={name} already has step={args.steps}; skipping', flush=True)
                final_rows.append(existing)
            else:
                rows = train_one(args, name, positions, seed, splits, cfg, device)
                final_rows.append(rows[-1])
            write_csv(args.out / 'final_gradient_rows_by_seed.csv', final_rows)
            agg = aggregate_final(final_rows)
            write_csv(args.out / 'aggregate_gradient_summary.csv', agg)
            write_summary(args.out / 'aggregate_gradient_summary.md', agg)
    print(args.out / 'aggregate_gradient_summary.md')


if __name__ == '__main__':
    main()
