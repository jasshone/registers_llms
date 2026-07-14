
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from run_realtext_toy_sink_training import (
    RealTextTinyGPT,
    RealTextToyConfig,
    encode_text,
    fixed_eval_batch,
    matched_random,
    read_tiny_shakespeare,
    read_wikitext,
    split_tokens,
)
from analyze_realtext_toy_checkpoints import (
    attention_qk_loss_chunked,
    mlp_scores_chunked,
    zero_eval_chunked,
)


def batched(batch: torch.Tensor, chunk_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, int(batch.shape[0]), chunk_size):
        yield batch[start : start + chunk_size]


def checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix('checkpoint_step'))


def load_tokens_from_run(run_dir: Path, token_cache: Path | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    cfg_json = json.loads((run_dir / 'config.json').read_text())
    run_args = cfg_json['args']
    if token_cache is not None and token_cache.exists():
        payload = torch.load(token_cache, map_location='cpu')
        if isinstance(payload, dict) and 'tokens' in payload:
            tokens = payload['tokens'].long().cpu()
            cache_meta = payload.get('metadata', {})
        elif torch.is_tensor(payload):
            tokens = payload.long().cpu()
            cache_meta = {}
        else:
            raise ValueError(f'Unsupported token cache format: {token_cache}')
        run_args = dict(run_args)
        run_args['token_cache'] = str(token_cache)
        run_args['token_cache_metadata'] = cache_meta
        return split_tokens(tokens, train_frac=0.9, val_frac=0.05), run_args
    if run_args['dataset'] == 'tiny_shakespeare':
        text = read_tiny_shakespeare(Path(run_args['data_path']))
    else:
        text = read_wikitext(run_args.get('wikitext_train_split', 'train'), run_args.get('max_chars'))
    max_chars = run_args.get('max_chars')
    if max_chars is not None:
        text = text[: int(max_chars)]
    tokens, _vocab_size, _bos_token_id, _encoding_source = encode_text(
        text,
        encoding=run_args.get('encoding', 'byte'),
        tokenizer_model_id=run_args.get('tokenizer_model_id', 'EleutherAI/pythia-70m'),
    )
    return split_tokens(tokens, train_frac=0.9, val_frac=0.05), run_args


def group_with_assignments(selected: torch.Tensor, num_dummy_tokens: int) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    by_layer: dict[int, list[int]] = {}
    for layer, neuron in selected.tolist():
        by_layer.setdefault(int(layer), []).append(int(neuron))
    grouped: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer, neurons in sorted(by_layer.items()):
        neurons_t = torch.tensor(sorted(neurons), dtype=torch.long)
        assignments = torch.arange(neurons_t.numel(), dtype=torch.long) % num_dummy_tokens
        grouped[layer] = (neurons_t, assignments)
    return grouped


@contextmanager
def patched_toy_mlp_reroute(
    model: RealTextTinyGPT,
    selected: torch.Tensor,
    *,
    num_dummy_tokens: int,
    relocation_scale: float,
    relocation_fraction: float,
    preserve_original: bool,
) -> Iterator[None]:
    if num_dummy_tokens < 1:
        raise ValueError('num_dummy_tokens must be at least 1')
    if not 0.0 <= relocation_fraction <= 1.0:
        raise ValueError('relocation_fraction must be in [0, 1]')
    originals: dict[int, Any] = {}
    dummy_start = 1
    try:
        for layer_idx, (neurons, assignments) in group_with_assignments(selected, num_dummy_tokens).items():
            mlp = model.blocks[layer_idx].mlp
            originals[layer_idx] = mlp.forward

            def make_forward(layer_mlp: torch.nn.Module, selected_neurons: torch.Tensor, dummy_assignments: torch.Tensor):
                def forward(
                    x: torch.Tensor,
                    *,
                    zero_bos_neurons: torch.Tensor | None = None,
                    capture: list[torch.Tensor] | None = None,
                ) -> torch.Tensor:
                    h = F.gelu(layer_mlp.c_fc(x))
                    if capture is not None:
                        capture.append(h if h.requires_grad else h.detach())
                    selected_on_device = selected_neurons.to(x.device)
                    assignments_on_device = dummy_assignments.to(x.device)
                    gate_selected = h[..., selected_on_device].clone()
                    original_positions = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                    original_positions[dummy_start : dummy_start + num_dummy_tokens] = False
                    original_selected = gate_selected[:, original_positions, :]
                    max_vals = original_selected.amax(dim=1)
                    if not preserve_original:
                        gate_selected[:, original_positions, :] = original_selected * (1.0 - relocation_fraction)
                    for dummy_idx in range(num_dummy_tokens):
                        assigned = assignments_on_device == dummy_idx
                        if assigned.any():
                            gate_selected[:, dummy_start + dummy_idx, assigned] = (
                                max_vals[:, assigned] * relocation_scale * relocation_fraction
                            )
                    h = h.clone()
                    h[..., selected_on_device] = gate_selected
                    if zero_bos_neurons is not None and zero_bos_neurons.numel() > 0:
                        h = h.clone()
                        h[:, 0, zero_bos_neurons.to(h.device)] = 0.0
                    return layer_mlp.c_proj(layer_mlp.dropout(h))

                return forward

            mlp.forward = make_forward(mlp, neurons, assignments)  # type: ignore[method-assign]
        yield
    finally:
        for layer_idx, original in originals.items():
            model.blocks[layer_idx].mlp.forward = original  # type: ignore[method-assign]


def build_dummy_after_bos_embeds(
    model: RealTextTinyGPT,
    batch: torch.Tensor,
    *,
    num_dummy_tokens: int,
    dummy_init: str,
) -> torch.Tensor:
    token_embeds = model.wte(batch)
    if num_dummy_tokens >= batch.shape[1] - 1:
        raise ValueError('Too many dummy tokens for sequence length')
    if dummy_init == 'zero':
        dummy = torch.zeros(
            (batch.shape[0], num_dummy_tokens, token_embeds.shape[-1]),
            dtype=token_embeds.dtype,
            device=token_embeds.device,
        )
    elif dummy_init == 'bos':
        dummy = token_embeds[:, :1, :].expand(-1, num_dummy_tokens, -1).clone()
    else:
        raise ValueError(f'Unsupported dummy_init: {dummy_init}')
    suffix_len = batch.shape[1] - 1 - num_dummy_tokens
    suffix = token_embeds[:, 1 : 1 + suffix_len, :]
    return torch.cat([token_embeds[:, :1, :], dummy, suffix], dim=1)


def content_nll(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, int]:
    loss = F.cross_entropy(
        logits.float().contiguous().view(-1, logits.shape[-1]),
        targets.contiguous().view(-1),
        reduction='sum',
    )
    return float(loss.detach().cpu()), int(targets.numel())


@torch.no_grad()
def relocation_eval_chunked(
    model: RealTextTinyGPT,
    batch: torch.Tensor,
    selected: torch.Tensor | None,
    *,
    chunk_size: int,
    num_dummy_tokens: int,
    dummy_init: str,
    relocation_scale: float,
    relocation_fraction: float,
    preserve_original: bool,
) -> dict[str, Any]:
    cfg = model.cfg
    suffix_len = batch.shape[1] - 1 - num_dummy_tokens
    clean_loss_sum = 0.0
    changed_loss_sum = 0.0
    token_count = 0
    clean_bos_layer = torch.zeros(cfg.n_layer, dtype=torch.float64)
    bos_after_layer = torch.zeros(cfg.n_layer, dtype=torch.float64)
    dummy_after_layer = torch.zeros(cfg.n_layer, dtype=torch.float64)
    clean_head = torch.zeros(cfg.n_layer, cfg.n_head, dtype=torch.float64)
    bos_after_head = torch.zeros(cfg.n_layer, cfg.n_head, dtype=torch.float64)
    dummy_after_head = torch.zeros(cfg.n_layer, cfg.n_head, dtype=torch.float64)
    examples = 0

    for chunk in batched(batch, chunk_size):
        targets = chunk[:, 1 : 1 + suffix_len]
        clean = model(chunk, output_attentions=True)
        clean_loss, ntok = content_nll(clean['logits'][:, :suffix_len, :], targets)
        embeds = build_dummy_after_bos_embeds(
            model,
            chunk,
            num_dummy_tokens=num_dummy_tokens,
            dummy_init=dummy_init,
        )
        if selected is None or selected.numel() == 0:
            changed = model(inputs_embeds=embeds, output_attentions=True)
        else:
            with patched_toy_mlp_reroute(
                model,
                selected,
                num_dummy_tokens=num_dummy_tokens,
                relocation_scale=relocation_scale,
                relocation_fraction=relocation_fraction,
                preserve_original=preserve_original,
            ):
                changed = model(inputs_embeds=embeds, output_attentions=True)
        changed_loss, _ = content_nll(
            changed['logits'][:, num_dummy_tokens : num_dummy_tokens + suffix_len, :],
            targets,
        )
        bsz = int(chunk.shape[0])
        examples += bsz
        clean_loss_sum += clean_loss
        changed_loss_sum += changed_loss
        token_count += ntok
        query_start = 1 + num_dummy_tokens
        for layer_idx, attn in enumerate(clean['attentions']):
            vals = attn[:, :, 1:, 0].float()
            clean_bos_layer[layer_idx] += vals.mean(dim=(1, 2)).sum().double().cpu()
            clean_head[layer_idx] += vals.mean(dim=2).sum(dim=0).double().cpu()
        for layer_idx, attn in enumerate(changed['attentions']):
            bos_vals = attn[:, :, query_start:, 0].float()
            dummy_vals = attn[:, :, query_start:, 1 : 1 + num_dummy_tokens].float()
            bos_after_layer[layer_idx] += bos_vals.mean(dim=(1, 2)).sum().double().cpu()
            dummy_after_layer[layer_idx] += dummy_vals.mean(dim=(1, 2)).sum(dim=-1).sum().double().cpu()
            bos_after_head[layer_idx] += bos_vals.mean(dim=2).sum(dim=0).double().cpu()
            dummy_after_head[layer_idx] += dummy_vals.mean(dim=2).sum(dim=-1).sum(dim=0).double().cpu()

    clean_nll = clean_loss_sum / max(1, token_count)
    changed_nll = changed_loss_sum / max(1, token_count)
    clean_bos_layer = (clean_bos_layer / max(1, examples)).float()
    bos_after_layer = (bos_after_layer / max(1, examples)).float()
    dummy_after_layer = (dummy_after_layer / max(1, examples)).float()
    clean_head = (clean_head / max(1, examples)).float()
    bos_after_head = (bos_after_head / max(1, examples)).float()
    dummy_after_head = (dummy_after_head / max(1, examples)).float()
    summary = dummy_after_layer - bos_after_layer
    return {
        'clean_content_nll': clean_nll,
        'changed_content_nll': changed_nll,
        'ppl_ratio': math.exp(changed_nll - clean_nll),
        'clean_bos_attention_mean': float(clean_bos_layer.mean()),
        'clean_bos_attention_max_head': float(clean_head.max()),
        'bos_attention_after_mean': float(bos_after_layer.mean()),
        'dummy_attention_after_mean': float(dummy_after_layer.mean()),
        'dummy_minus_bos_mean': float(summary.mean()),
        'dummy_minus_bos_max_layer': float(summary.max()),
        'dummy_attention_after_max_head': float(dummy_after_head.max()),
        'bos_attention_after_max_head': float(bos_after_head.max()),
        'clean_bos_attention_by_layer': ' '.join(f'{float(x):.6g}' for x in clean_bos_layer.tolist()),
        'dummy_attention_after_by_layer': ' '.join(f'{float(x):.6g}' for x in dummy_after_layer.tolist()),
        'bos_attention_after_by_layer': ' '.join(f'{float(x):.6g}' for x in bos_after_layer.tolist()),
        'dummy_minus_bos_by_layer': ' '.join(f'{float(x):.6g}' for x in summary.tolist()),
    }


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
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(val):
        return 'nan'
    return f'{val:.{digits}g}'


def write_md(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    final = rows[-1] if rows else {}
    positive = [r for r in rows if float(r['selected_dummy_minus_bos_mean']) > 0]
    first_positive = min(positive, key=lambda r: int(r['step'])) if positive else None
    strongest = max(rows, key=lambda r: float(r['selected_dummy_minus_bos_mean'])) if rows else None
    lines = [
        '# Toy-to-Pretrained Relocation Bridge',
        '',
        'Question: in toy checkpoints, do source/sink-selective MLP neurons appear and become sufficient to move attention to a dummy/register position, as in the pretrained Pythia relocation experiments?',
        '',
        'Setup: select top-k MLP neurons by BOS-position activation excess on train windows. On held-out windows, insert a zero dummy immediately after BOS, patch the selected MLP activations into that dummy using the same max-over-source-positions relocation rule used for pretrained models, and compare against layer-matched random neurons.',
        '',
        f"Run dir: `{config['run_dir']}`",
        f"Eval split/examples: `{config['eval_split']}` / `{config['eval_examples']}`",
        f"Top-k / random controls: `{config['topk']}` / `{config['random_controls']}`",
        f"Dummy tokens / scale / fraction: `{config['num_dummy_tokens']}` / `{config['relocation_scale']}` / `{config['relocation_fraction']}`",
        '',
    ]
    if first_positive:
        lines.append(f"First checkpoint with positive selected dummy-minus-BOS relocation: step `{first_positive['step']}` ({fmt(first_positive['selected_dummy_minus_bos_mean'])}).")
    if strongest:
        lines.append(f"Strongest selected relocation: step `{strongest['step']}` with dummy-minus-BOS `{fmt(strongest['selected_dummy_minus_bos_mean'])}` versus random `{fmt(strongest['random_dummy_minus_bos_mean'])}` and dummy-only `{fmt(strongest['dummy_only_dummy_minus_bos_mean'])}`.")
    if final:
        lines.append(f"Final checkpoint: clean max-head BOS attention `{fmt(final['clean_max_head_bos_attention'])}`, selected score mass `{fmt(final['selected_score_mass'])}`, selected dummy-minus-BOS `{fmt(final['selected_dummy_minus_bos_mean'])}`, random dummy-minus-BOS `{fmt(final['random_dummy_minus_bos_mean'])}`, selected PPL ratio `{fmt(final['selected_ppl_ratio'])}`.")
    lines.extend([
        '',
        '| step | clean max BOS | score mass | zero BOS ret | random zero ret | dummy-only d-BOS | selected d-BOS | random d-BOS | selected PPL | random PPL |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for r in rows:
        lines.append(
            f"| {r['step']} | {fmt(r['clean_max_head_bos_attention'])} | {fmt(r['selected_score_mass'])} | "
            f"{fmt(r['zero_max_head_bos_retention'])} | {fmt(r['random_zero_max_head_bos_retention'])} | "
            f"{fmt(r['dummy_only_dummy_minus_bos_mean'])} | {fmt(r['selected_dummy_minus_bos_mean'])} | "
            f"{fmt(r['random_dummy_minus_bos_mean'])} | {fmt(r['selected_ppl_ratio'])} | {fmt(r['random_ppl_ratio_mean'])} |"
        )
    lines.extend([
        '',
        'Paper-safe interpretation: this bridges the toy and pretrained results if selected toy MLP features become more source-selective over training, ablation reduces BOS attention more than random controls, and the same selected features are sufficient to make a dummy/register slot compete for sink attention. It does not prove the toy and Pythia circuits are identical; it shows the same observable causal motif can arise in the controlled toy setting.',
    ])
    path.write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Toy-to-pretrained bridge: checkpoint-wise selected-neuron relocation in real-text toy LM.')
    p.add_argument('--run-dir', type=Path, default=Path('outputs/realtext_toy_sink_training/wikitext_pythia_tokenizer_10k_eval128'))
    p.add_argument('--token-cache', type=Path, default=Path('outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt'))
    p.add_argument('--out', type=Path, default=Path('outputs/toy_pretrained_bridge/wikitext_pythia_tokenizer_10k'))
    p.add_argument('--eval-split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--selection-examples', type=int, default=512)
    p.add_argument('--eval-examples', type=int, default=512)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--topk', type=int, default=32)
    p.add_argument('--random-controls', type=int, default=5)
    p.add_argument('--num-dummy-tokens', type=int, default=1)
    p.add_argument('--dummy-init', choices=['zero', 'bos'], default='zero')
    p.add_argument('--relocation-scale', type=float, default=1.0)
    p.add_argument('--relocation-fraction', type=float, default=1.0)
    p.add_argument('--preserve-original', action='store_true')
    p.add_argument('--steps', nargs='*', type=int, default=None)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    splits, run_args = load_tokens_from_run(args.run_dir, args.token_cache)
    ckpts = sorted(args.run_dir.glob('checkpoint_step*.pt'), key=checkpoint_step)
    if args.steps is not None:
        keep = set(args.steps)
        ckpts = [p for p in ckpts if checkpoint_step(p) in keep]
    if not ckpts:
        raise ValueError(f'no checkpoints found in {args.run_dir}')
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for ckpt_path in ckpts:
        payload = torch.load(ckpt_path, map_location='cpu')
        cfg = RealTextToyConfig(**payload['config'])
        model = RealTextTinyGPT(cfg).to(device)
        model.load_state_dict(payload['model'])
        model.eval()
        step = int(payload['step'])
        selection_batch = fixed_eval_batch(splits['train'], cfg, args.selection_examples, device)
        eval_batch = fixed_eval_batch(splits[args.eval_split], cfg, args.eval_examples, device)
        scores = mlp_scores_chunked(model, selection_batch, args.chunk_size)
        flat = scores.flatten()
        vals, idx = torch.topk(flat, k=min(args.topk, flat.numel()))
        selected = torch.stack([idx // scores.shape[1], idx % scores.shape[1]], dim=1).long()
        attention_summary = attention_qk_loss_chunked(model, eval_batch, args.chunk_size)
        zero = zero_eval_chunked(model, eval_batch, selected.to(device), args.chunk_size)
        dummy_only = relocation_eval_chunked(
            model,
            eval_batch,
            None,
            chunk_size=args.chunk_size,
            num_dummy_tokens=args.num_dummy_tokens,
            dummy_init=args.dummy_init,
            relocation_scale=args.relocation_scale,
            relocation_fraction=args.relocation_fraction,
            preserve_original=args.preserve_original,
        )
        selected_reloc = relocation_eval_chunked(
            model,
            eval_batch,
            selected.to(device),
            chunk_size=args.chunk_size,
            num_dummy_tokens=args.num_dummy_tokens,
            dummy_init=args.dummy_init,
            relocation_scale=args.relocation_scale,
            relocation_fraction=args.relocation_fraction,
            preserve_original=args.preserve_original,
        )
        random_relocs = []
        random_zero_rets = []
        random_zero_ppls = []
        for ridx in range(args.random_controls):
            rnd = matched_random(selected, cfg, seed=2_000_000 + step * 100 + ridx)
            rz = zero_eval_chunked(model, eval_batch, rnd.to(device), args.chunk_size)
            rr = relocation_eval_chunked(
                model,
                eval_batch,
                rnd.to(device),
                chunk_size=args.chunk_size,
                num_dummy_tokens=args.num_dummy_tokens,
                dummy_init=args.dummy_init,
                relocation_scale=args.relocation_scale,
                relocation_fraction=args.relocation_fraction,
                preserve_original=args.preserve_original,
            )
            random_relocs.append(rr)
            random_zero_rets.append(rz['zero_max_head_bos_retention'])
            random_zero_ppls.append(rz['zero_ppl_ratio'])
        random_dummy_minus = [float(r['dummy_minus_bos_mean']) for r in random_relocs]
        random_ppl = [float(r['ppl_ratio']) for r in random_relocs]
        random_dummy_attn = [float(r['dummy_attention_after_mean']) for r in random_relocs]
        random_bos_attn = [float(r['bos_attention_after_mean']) for r in random_relocs]
        row: dict[str, Any] = {
            'step': step,
            'eval_split': args.eval_split,
            'selection_split': 'train',
            'selection_examples': args.selection_examples,
            'eval_examples': args.eval_examples,
            'chunk_size': args.chunk_size,
            'topk': args.topk,
            'random_controls': args.random_controls,
            'num_dummy_tokens': args.num_dummy_tokens,
            'dummy_init': args.dummy_init,
            'relocation_scale': args.relocation_scale,
            'relocation_fraction': args.relocation_fraction,
            'preserve_original': args.preserve_original,
            'selected_score_mass': float(vals.clamp_min(0).sum()),
            'selected_score_mean': float(vals.mean()),
            'selected_layers': ' '.join(str(int(x)) for x in selected[:, 0].tolist()),
            'zero_ppl_ratio': zero['zero_ppl_ratio'],
            'zero_mean_bos_retention': zero['zero_mean_bos_retention'],
            'zero_max_head_bos_retention': zero['zero_max_head_bos_retention'],
            'random_zero_ppl_ratio_mean': mean(random_zero_ppls) if random_zero_ppls else float('nan'),
            'random_zero_max_head_bos_retention': mean(random_zero_rets) if random_zero_rets else float('nan'),
            'dummy_only_dummy_minus_bos_mean': dummy_only['dummy_minus_bos_mean'],
            'dummy_only_dummy_attention_after_mean': dummy_only['dummy_attention_after_mean'],
            'dummy_only_bos_attention_after_mean': dummy_only['bos_attention_after_mean'],
            'dummy_only_ppl_ratio': dummy_only['ppl_ratio'],
            'selected_dummy_minus_bos_mean': selected_reloc['dummy_minus_bos_mean'],
            'selected_dummy_minus_bos_max_layer': selected_reloc['dummy_minus_bos_max_layer'],
            'selected_dummy_attention_after_mean': selected_reloc['dummy_attention_after_mean'],
            'selected_bos_attention_after_mean': selected_reloc['bos_attention_after_mean'],
            'selected_dummy_attention_after_max_head': selected_reloc['dummy_attention_after_max_head'],
            'selected_bos_attention_after_max_head': selected_reloc['bos_attention_after_max_head'],
            'selected_relocation_lift_vs_dummy_only': selected_reloc['dummy_minus_bos_mean'] - dummy_only['dummy_minus_bos_mean'],
            'selected_ppl_ratio': selected_reloc['ppl_ratio'],
            'random_dummy_minus_bos_mean': mean(random_dummy_minus) if random_dummy_minus else float('nan'),
            'random_dummy_minus_bos_std': stdev(random_dummy_minus) if len(random_dummy_minus) > 1 else 0.0,
            'random_dummy_attention_after_mean': mean(random_dummy_attn) if random_dummy_attn else float('nan'),
            'random_bos_attention_after_mean': mean(random_bos_attn) if random_bos_attn else float('nan'),
            'random_ppl_ratio_mean': mean(random_ppl) if random_ppl else float('nan'),
            'random_ppl_ratio_std': stdev(random_ppl) if len(random_ppl) > 1 else 0.0,
            'selected_dummy_minus_bos_by_layer': selected_reloc['dummy_minus_bos_by_layer'],
            'random_dummy_minus_bos_values': ' '.join(f'{x:.6g}' for x in random_dummy_minus),
        }
        row.update({
            'clean_loss': attention_summary['loss'],
            'clean_ppl': attention_summary['ppl'],
            'clean_mean_bos_attention': attention_summary['mean_bos_attention'],
            'clean_max_head_bos_attention': attention_summary['max_head_bos_attention'],
            'clean_max_head_bos_layer': attention_summary['max_head_bos_layer'],
            'clean_max_head_bos_head': attention_summary['max_head_bos_head'],
            'clean_qk_margin_at_max_head': attention_summary['qk_margin_at_max_head'],
            'clean_max_qk_margin': attention_summary['max_qk_margin'],
        })
        rows.append(row)
        print(
            f"[row] step={step} clean_max_bos={row['clean_max_head_bos_attention']:.4f} "
            f"score={row['selected_score_mass']:.4g} selected_d-bos={row['selected_dummy_minus_bos_mean']:+.4f} "
            f"random_d-bos={row['random_dummy_minus_bos_mean']:+.4f} zero_ret={row['zero_max_head_bos_retention']:.3f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_csv(args.out / 'toy_relocation_bridge_metrics.csv', rows)
        write_md(args.out / 'toy_relocation_bridge_summary.md', rows, vars(args) | {'run_dir': str(args.run_dir)})
    (args.out / 'config.json').write_text(json.dumps({'args': vars(args), 'source_run_args': run_args}, indent=2, default=str) + '\n')
    print(args.out / 'toy_relocation_bridge_summary.md')


if __name__ == '__main__':
    main()
