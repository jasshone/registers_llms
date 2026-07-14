
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from run_realtext_toy_bos_presence_controls import parse_variant, train_variant, write_csv
from run_realtext_toy_sink_training import RealTextToyConfig, split_tokens


def load_token_cache(path: Path, max_tokens: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict) or 'tokens' not in payload:
        raise ValueError(f'{path} must contain a dict with a tokens tensor')
    tokens = payload['tokens'].long()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens, dict(payload.get('metadata', {}))


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else float('nan')


def std(xs: list[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for row in rows:
        if row['variant'] not in variants:
            variants.append(row['variant'])
    numeric = [
        'loss', 'ppl', 'pos0_attention_mean', 'control_attention_mean',
        'pos0_minus_control_attention', 'bos_attention_sum_mean',
        'bos_attention_max_position_mean', 'bos_attention_max_head_over_positions',
    ]
    out = []
    for variant in variants:
        group = [r for r in rows if r['variant'] == variant]
        agg: dict[str, Any] = {
            'variant': variant,
            'bos_positions': group[0]['bos_positions'],
            'seeds': ' '.join(str(r['seed']) for r in group),
            'n_seeds': len(group),
        }
        for key in numeric:
            vals = [float(r[key]) for r in group if key in r and math.isfinite(float(r[key]))]
            agg[f'{key}_mean'] = mean(vals)
            agg[f'{key}_std'] = std(vals)
        out.append(agg)
    return out


def write_aggregate_md(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        '# Toy BOS Presence Controls on Wikitext Train Text',
        '',
        'Toy LMs trained from scratch on actual Wikitext-103 train text tokenized with the Pythia-70M tokenizer. Each row aggregates final-step metrics across seeds.',
        '',
        f"Token source: `{metadata.get('dataset', 'unknown')}` / `{metadata.get('config', 'unknown')}` / split `{metadata.get('split', 'unknown')}`; max chars `{metadata.get('max_chars', 'unknown')}`; tokens used `{metadata.get('tokens_used', 'unknown')}`.",
        '',
        '| variant | BOS positions | seeds | loss | PPL | pos0 attn | control attn | pos0-control | BOS attn sum | max BOS-pos attn |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['bos_positions']} | {r['n_seeds']} | "
            f"{r['loss_mean']:.4f} +/- {r['loss_std']:.4f} | "
            f"{r['ppl_mean']:.3g} +/- {r['ppl_std']:.3g} | "
            f"{r['pos0_attention_mean_mean']:.4f} +/- {r['pos0_attention_mean_std']:.4f} | "
            f"{r['control_attention_mean_mean']:.4f} +/- {r['control_attention_mean_std']:.4f} | "
            f"{r['pos0_minus_control_attention_mean']:+.4f} +/- {r['pos0_minus_control_attention_std']:.4f} | "
            f"{r['bos_attention_sum_mean_mean']:.4f} +/- {r['bos_attention_sum_mean_std']:.4f} | "
            f"{r['bos_attention_max_position_mean_mean']:.4f} +/- {r['bos_attention_max_position_mean_std']:.4f} |"
        )
    lines += [
        '',
        'Interpretation guide: `no_bos` tests whether ordinary text at absolute position 0 becomes sink-like. `single_8` and `single_32` test whether the sink follows BOS identity. Duplicate variants test whether later BOS copies compete with the first-position BOS.',
    ]
    path.write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run 3-seed toy BOS controls on actual Wikitext train-token cache.')
    ap.add_argument('--token-cache', type=Path, default=Path('outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt'))
    ap.add_argument('--out', type=Path, default=Path('outputs/realtext_toy_bos_presence_controls/wikitext103_train_3seed'))
    ap.add_argument('--variants', nargs='*', default=['single_0', 'no_bos', 'single_8', 'single_32', 'duplicate_0,8', 'duplicate_0,32'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--max-tokens', type=int, default=None)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--eval-batch-size', type=int, default=128)
    ap.add_argument('--eval-interval', type=int, default=1000)
    ap.add_argument('--eval-split', choices=['val', 'test'], default='val')
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--n-layer', type=int, default=4)
    ap.add_argument('--n-head', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--d-mlp', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device if args.device != 'auto' else 'cpu')
    tokens, metadata = load_token_cache(args.token_cache, args.max_tokens)
    metadata['tokens_used'] = int(tokens.numel())
    splits = split_tokens(tokens, train_frac=0.9, val_frac=0.05)
    vocab_size = int(metadata.get('vocab_size', 50277))
    bos_token_id = int(metadata.get('bos_token_id', 0))
    cfg = RealTextToyConfig(vocab_size=vocab_size, bos_token_id=bos_token_id, seq_len=args.seq_len, n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model, d_mlp=args.d_mlp)
    variants = [parse_variant(v) for v in args.variants]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps({'args': vars(args), 'model': asdict(cfg), 'metadata': metadata, 'split_tokens': {k: int(v.numel()) for k, v in splits.items()}, 'variants': [{'name': n, 'bos_positions': ps} for n, ps in variants]}, indent=2, default=str) + '\n')
    all_final = []
    for seed in args.seeds:
        seed_out = args.out / f'seed_{seed}'
        seed_args = SimpleNamespace(**vars(args))
        seed_args.out = seed_out
        seed_args.seed = seed
        for name, positions in variants:
            final = train_variant(seed_args, name, positions, splits, cfg, device)
            final['seed'] = seed
            all_final.append(final)
            write_csv(args.out / 'final_rows_by_seed.csv', all_final)
            agg = aggregate_rows(all_final)
            write_csv(args.out / 'aggregate_summary.csv', agg)
            write_aggregate_md(args.out / 'aggregate_summary.md', agg, metadata)
    print(args.out / 'aggregate_summary.md')


if __name__ == '__main__':
    main()
