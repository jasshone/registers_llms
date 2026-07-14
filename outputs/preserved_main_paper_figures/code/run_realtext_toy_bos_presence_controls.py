
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from run_realtext_toy_sink_training import RealTextTinyGPT, RealTextToyConfig, split_tokens


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


def load_token_windows(path: Path, max_tokens: int | None) -> torch.Tensor:
    payload = torch.load(path, map_location='cpu')
    windows = payload['windows'] if isinstance(payload, dict) else payload
    if windows.ndim != 2 or windows.shape[1] < 2:
        raise ValueError(f'expected 2D token windows, got {tuple(windows.shape)}')
    # Drop each cached window's artificial leading start token and use the rest as ordinary text.
    tokens = windows[:, 1:].contiguous().view(-1).long()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def parse_variant(text: str) -> tuple[str, list[int]]:
    if text == 'no_bos':
        return text, []
    if text.startswith('single_'):
        return text, [int(text.split('_', 1)[1])]
    if text.startswith('duplicate_'):
        return text, [int(x) for x in text.split('_', 1)[1].split(',')]
    raise ValueError(f'unknown variant {text!r}; use no_bos, single_POS, or duplicate_POS,POS')


def make_variant_batch(
    tokens: torch.Tensor,
    cfg: RealTextToyConfig,
    batch_size: int,
    device: torch.device,
    *,
    bos_positions: list[int],
    fixed: bool = False,
) -> torch.Tensor:
    bos_positions = sorted(set(int(p) for p in bos_positions))
    if any(p < 0 or p >= cfg.seq_len for p in bos_positions):
        raise ValueError(f'bos positions must be in [0, {cfg.seq_len - 1}], got {bos_positions}')
    content_len = cfg.seq_len - len(bos_positions)
    max_start = int(tokens.numel()) - content_len
    if max_start <= 0:
        raise ValueError('token stream too short')
    if fixed:
        starts = [0] if batch_size == 1 else torch.linspace(0, max_start - 1, steps=batch_size).long().tolist()
    else:
        starts = torch.randint(0, max_start, (batch_size,)).tolist()
    bos_set = set(bos_positions)
    rows = []
    for start in starts:
        content = tokens[int(start): int(start) + content_len]
        row = torch.empty(cfg.seq_len, dtype=torch.long)
        content_idx = 0
        for pos in range(cfg.seq_len):
            if pos in bos_set:
                row[pos] = cfg.bos_token_id
            else:
                row[pos] = content[content_idx]
                content_idx += 1
        rows.append(row)
    return torch.stack(rows, dim=0).to(device)


def mean_attention_to_key(attentions: list[torch.Tensor], key_pos: int, *, query_start: int | None = None) -> tuple[float, float, str]:
    by_layer = []
    by_head = []
    for attn in attentions:
        start = key_pos + 1 if query_start is None else max(query_start, key_pos + 1)
        if start >= attn.shape[-2]:
            layer_vals = torch.full((attn.shape[1],), float('nan'), device=attn.device)
        else:
            layer_vals = attn[:, :, start:, key_pos].float().mean(dim=(0, 2))
        by_head.append(layer_vals.detach().cpu())
        by_layer.append(layer_vals.mean().detach().cpu())
    by_layer_t = torch.stack(by_layer)
    by_head_t = torch.stack(by_head)
    finite = by_head_t[torch.isfinite(by_head_t)]
    max_head = float(finite.max()) if finite.numel() else float('nan')
    return float(torch.nanmean(by_layer_t)), max_head, ' '.join(f'{float(x):.6g}' for x in by_layer_t.tolist())


def summarize_attention(attentions: list[torch.Tensor], bos_positions: list[int], seq_len: int) -> dict[str, Any]:
    pos0_mean, pos0_max, pos0_layers = mean_attention_to_key(attentions, 0)
    ordinary_control = 1 if 1 not in bos_positions and seq_len > 2 else min(2, seq_len - 1)
    control_mean, control_max, control_layers = mean_attention_to_key(attentions, ordinary_control)
    row: dict[str, Any] = {
        'pos0_attention_mean': pos0_mean,
        'pos0_attention_max_head': pos0_max,
        'control_position': ordinary_control,
        'control_attention_mean': control_mean,
        'control_attention_max_head': control_max,
        'pos0_minus_control_attention': pos0_mean - control_mean,
        'pos0_attention_by_layer': pos0_layers,
        'control_attention_by_layer': control_layers,
    }
    bos_means = []
    bos_maxes = []
    bos_layers = []
    for pos in bos_positions:
        m, mx, layers = mean_attention_to_key(attentions, pos)
        row[f'bos_pos{pos}_attention_mean'] = m
        row[f'bos_pos{pos}_attention_max_head'] = mx
        row[f'bos_pos{pos}_attention_by_layer'] = layers
        bos_means.append(m)
        bos_maxes.append(mx)
        bos_layers.append(layers)
    row['bos_attention_sum_mean'] = float(sum(bos_means)) if bos_means else 0.0
    row['bos_attention_max_position_mean'] = float(max(bos_means)) if bos_means else float('nan')
    row['bos_attention_max_head_over_positions'] = float(max(bos_maxes)) if bos_maxes else float('nan')
    return row


def train_variant(args: argparse.Namespace, name: str, bos_positions: list[int], splits: dict[str, torch.Tensor], cfg: RealTextToyConfig, device: torch.device) -> dict[str, Any]:
    seed = args.seed + sum((idx + 1) * (pos + 17) for idx, pos in enumerate(bos_positions)) + 7919 * len(bos_positions)
    torch.manual_seed(seed)
    random.seed(seed)
    model = RealTextTinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_batch = make_variant_batch(splits[args.eval_split], cfg, args.eval_batch_size, device, bos_positions=bos_positions, fixed=True)
    rows: list[dict[str, Any]] = []
    out_dir = args.out / name.replace(',', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                out = model(eval_batch, output_attentions=True)
                loss = F.cross_entropy(out['logits'][:, :-1, :].contiguous().view(-1, cfg.vocab_size), eval_batch[:, 1:].contiguous().view(-1))
            row = {
                'variant': name,
                'bos_positions': ' '.join(str(p) for p in bos_positions) if bos_positions else 'none',
                'step': step,
                'loss': float(loss.detach().cpu()),
                'ppl': float(torch.exp(loss).detach().cpu()),
            }
            row.update(summarize_attention(out['attentions'], bos_positions, cfg.seq_len))
            rows.append(row)
            write_csv(out_dir / 'metrics.csv', rows)
            print(f"[{name}] step={step} loss={row['loss']:.4f} pos0={row['pos0_attention_mean']:.4f} ctrl={row['control_attention_mean']:.4f} bos_sum={row['bos_attention_sum_mean']:.4f}", flush=True)
            model.train()
        if step == args.steps:
            break
        batch = make_variant_batch(splits['train'], cfg, args.batch_size, device, bos_positions=bos_positions, fixed=False)
        out = model(batch, labels=batch)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return rows[-1]


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        '# Toy BOS Presence Controls',
        '',
        'Fresh toy LMs trained on the same cached Pythia-tokenized Wikitext windows. Variants test whether sink attention requires a BOS token at position 0, any BOS token, or merely an early globally visible position.',
        '',
        '| variant | BOS positions | final loss | PPL | pos0 attn | control attn | pos0-control | BOS attn sum | max BOS-position attn |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['bos_positions']} | {r['loss']:.4f} | {r['ppl']:.3g} | "
            f"{r['pos0_attention_mean']:.4f} | {r['control_attention_mean']:.4f} | {r['pos0_minus_control_attention']:+.4f} | "
            f"{r['bos_attention_sum_mean']:.4f} | {r['bos_attention_max_position_mean']:.4f} |"
        )
    path.write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train toy LMs under BOS absent/shifted/duplicated controls.')
    p.add_argument('--token-windows', type=Path, default=Path('outputs/final_split_bias_transfer_official_val/pythia_70m/wikitext103_validation_windows_full_1024.pt'))
    p.add_argument('--variants', nargs='*', default=['single_0', 'no_bos', 'single_8', 'duplicate_0,8', 'duplicate_0,32'])
    p.add_argument('--out', type=Path, default=Path('outputs/realtext_toy_bos_presence_controls/wikitext_pythia_cached_val_windows'))
    p.add_argument('--max-tokens', type=int, default=10_000_000)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--eval-interval', type=int, default=1000)
    p.add_argument('--eval-split', choices=['val', 'test'], default='val')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-head', type=int, default=4)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--d-mlp', type=int, default=512)
    p.add_argument('--vocab-size', type=int, default=50277)
    p.add_argument('--bos-token-id', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    tokens = load_token_windows(args.token_windows, args.max_tokens)
    splits = split_tokens(tokens, train_frac=0.9, val_frac=0.05)
    cfg = RealTextToyConfig(vocab_size=args.vocab_size, bos_token_id=args.bos_token_id, seq_len=args.seq_len, n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model, d_mlp=args.d_mlp)
    args.out.mkdir(parents=True, exist_ok=True)
    variants = [parse_variant(v) for v in args.variants]
    (args.out / 'config.json').write_text(json.dumps({'args': vars(args), 'model': asdict(cfg), 'num_tokens': int(tokens.numel()), 'split_tokens': {k: int(v.numel()) for k, v in splits.items()}, 'variants': [{'name': n, 'bos_positions': ps} for n, ps in variants]}, indent=2, default=str) + '\n')
    final_rows = []
    for name, positions in variants:
        final_rows.append(train_variant(args, name, positions, splits, cfg, device))
        write_csv(args.out / 'bos_presence_control_summary.csv', final_rows)
        write_summary(args.out / 'bos_presence_control_summary.md', final_rows)
    print(args.out / 'bos_presence_control_summary.md')


if __name__ == '__main__':
    main()
