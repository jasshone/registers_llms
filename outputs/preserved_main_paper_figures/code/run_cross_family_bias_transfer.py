from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from run_autoresearch_sink_mechanism import (
    BATCH_SIZE,
    clean_baseline,
    emergence_layers,
    evaluate_bias_method,
    layer_counts,
    load_or_compute_sink_mask,
    make_masks,
    score_candidate_neurons,
)
from sink_neurons.env import configure_runtime
from sink_neurons.modeling import load_model
from sink_neurons.windows import load_windows_artifact

OUT = Path('outputs/cross_family_bias_transfer')
MAX_WINDOWS = 32
TOPKS = [1, 2, 4, 8, 16]
SCALES = [0.9, 1.0, 1.25, 1.5, 2.0, 3.0]
MASKS = ['bos_only', 'pipeline_sinks_plus_bos']

@dataclass(frozen=True)
class Cfg:
    key: str
    family: str
    model_id: str
    windows: Path
    score_artifact: Path | None = None
    use_cached_sink_mask: bool = True

CONFIGS = [
    Cfg('gpt2_small', 'gpt2', 'gpt2', Path('outputs/gpt2_confirmation/windows_128x1023.pt'), Path('outputs/gpt2_confirmation/scores.pt')),
    Cfg('gpt2_medium', 'gpt2', 'gpt2-medium', Path('outputs/gpt2_confirmation/windows_128x1023.pt'), Path('outputs/gpt2_medium/same_method_scale10_bos_ppl/scores.pt'), False),
    Cfg('pythia_1b', 'pythia', 'EleutherAI/pythia-1b', Path('outputs/pythia_1b/windows_128x1024.pt'), Path('outputs/pythia_1b/attention_sink_register_neurons_percentile_s5/scores.pt')),
    Cfg('qwen3_1_7b', 'qwen3', 'Qwen/Qwen3-1.7B', Path('outputs/qwen3_1_7b/windows_32x1024.pt'), Path('outputs/qwen3_1_7b/attention_sink_register_neurons_percentile_32_include_pos0/scores.pt')),
    Cfg('qwen2_5_1_5b', 'qwen2.5', 'Qwen/Qwen2.5-1.5B', Path('outputs/qwen2_5_1_5b/windows_128x1024.pt'), Path('outputs/qwen2_5_1_5b/attention_sink_register_neurons_percentile_include_pos0/scores.pt')),
    Cfg('llama3_8b', 'llama3', 'meta-llama/Meta-Llama-3-8B', Path('outputs/llama3_8b/windows_128x1024.pt'), Path('outputs/llama3_8b/attention_sink_register_neurons_512/scores.pt')),
    Cfg('gemma2_2b', 'gemma2', 'google/gemma-2-2b', Path('outputs/gemma2_2b/windows_128x1024.pt'), Path('outputs/gemma2_2b/attention_sink_register_neurons_percentile_include_pos0/scores.pt')),
    Cfg('phi_2', 'phi', 'microsoft/phi-2', Path('outputs/phi_mistral_episode_rule/phi_2/windows_32x1024.pt'), Path('outputs/phi_mistral_episode_rule/phi_2/attention_sink_register_neurons_percentile_32_include_pos0/scores.pt')),
    Cfg('mistral_7b_v0_1', 'mistral', 'mistralai/Mistral-7B-v0.1', Path('outputs/phi_mistral_episode_rule/mistral_7b_v0_1/windows_32x1024.pt'), Path('outputs/phi_mistral_episode_rule/mistral_7b_v0_1/attention_sink_register_neurons_percentile_32_include_pos0/scores.pt')),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    preferred = ['model_key','family','model_id','mask','mode','scale','topk','neurons','layer_counts','candidate_layers','episodes','sink_positions','bos_before_mean','bos_after_mean','dummy_after_mean','dummy_minus_bos_mean','baseline_ppl','intervened_ppl','ppl_ratio','delta_mean_nll','error']
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cfg_adapter(cfg: Cfg):
    class Adapter:
        key = cfg.key
        model_id = cfg.model_id
        windows = cfg.windows
        score_artifact = cfg.score_artifact
        use_cached_sink_mask = cfg.use_cached_sink_mask
    return Adapter()


def run_one(cfg: Cfg) -> list[dict]:
    out_dir = OUT / cfg.key
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    windows, _ = load_windows_artifact(cfg.windows)
    windows = windows[:MAX_WINDOWS]
    model = load_model(cfg.model_id, eager_attention=True)
    clean = clean_baseline(model, windows)
    cand_layers = emergence_layers(clean['bos_attention'])
    sink_mask = load_or_compute_sink_mask(cfg_adapter(cfg), model, windows, out_dir)
    masks = make_masks(windows, sink_mask)
    context = {
        'model_key': cfg.key,
        'family': cfg.family,
        'model_id': cfg.model_id,
        'windows': int(windows.shape[0]),
        'mean_bos_attention': clean['mean_bos'],
        'max_layer': clean['max_layer'],
        'max_bos_attention': clean['max_bos'],
        'episodes': clean['episodes'],
        'candidate_layers': cand_layers,
        'sink_positions': int(sink_mask.sum().item()),
        'method': 'candidate_layers = emergence band from clean BOS-attention curve; neuron_score = mean_gate(source_mask) - mean_gate(non_source_non_bos_control) + 0.1*abs(mean_gate(source_mask)); transfer subtracts source bias and writes same bias to zero dummy.',
    }
    (out_dir / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    print(json.dumps({'event':'context', **context}), flush=True)

    for mask_name in MASKS:
        mask = masks[mask_name]
        cdir = out_dir / mask_name
        cand = score_candidate_neurons(model, windows, mask, cand_layers, cdir)['selected_top']
        for topk in TOPKS:
            selected = cand[:topk]
            if selected.numel() == 0:
                continue
            # Controls at scale 1, then transfer scale sweep.
            specs = [('subtract_only', 1.0), ('add_only', 1.0)] + [('transfer', s) for s in SCALES]
            for mode, scale in specs:
                row = {
                    'model_key': cfg.key,
                    'family': cfg.family,
                    'model_id': cfg.model_id,
                    'mask': mask_name,
                    'mode': mode,
                    'scale': scale,
                    'topk': topk,
                    'neurons': int(selected.shape[0]),
                    'layer_counts': layer_counts(selected),
                    'candidate_layers': ' '.join(str(x) for x in cand_layers),
                    'episodes': json.dumps(clean['episodes']),
                    'sink_positions': int(sink_mask.sum().item()),
                }
                try:
                    row.update(evaluate_bias_method(model, windows, selected, mask, mode, scale))
                except Exception as exc:
                    row['error'] = repr(exc)
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_csv(out_dir / 'bias_transfer_sweep.csv', rows)
                all_existing = []
                for p in sorted(OUT.glob('*/bias_transfer_sweep.csv')):
                    with p.open() as f:
                        all_existing.extend(csv.DictReader(f))
                write_csv(OUT / 'summary.csv', all_existing)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize() -> None:
    rows = read_rows(OUT / 'summary.csv')
    by_model = {}
    for r in rows:
        if r.get('error'):
            continue
        by_model.setdefault(r['model_key'], []).append(r)
    lines = ['# Cross-Family Zero-Dummy Bias-Transfer Sweep', '']
    lines.append('All rows use the same neuron-finding method: infer an emergence band from the clean BOS-attention curve, score MLP neurons in that band by source-token gate excess over controls, then evaluate zero-dummy source-to-dummy bias transfer. This is new bias-transfer, not the older max-relocation intervention.')
    lines.append('')
    lines.append('| model | family | best compact row | dummy-BOS | PPL ratio | notes |')
    lines.append('| --- | --- | --- | ---: | ---: | --- |')
    for model_key, rs in sorted(by_model.items()):
        transfers = [r for r in rs if r['mode'] == 'transfer']
        strong = [r for r in transfers if float(r['dummy_minus_bos_mean']) > 0.30]
        if strong:
            best = min(strong, key=lambda r: (float(r['ppl_ratio']), int(r['neurons'])))
            note = 'strong relocation threshold met'
        else:
            best = max(transfers, key=lambda r: float(r['dummy_minus_bos_mean'])) if transfers else None
            note = 'threshold not met; showing max relocation'
        if best is None:
            continue
        rowdesc = f"{best['mask']} k={best['topk']} scale={float(best['scale']):g} layers={best['layer_counts']}"
        lines.append(f"| {model_key} | {best['family']} | `{rowdesc}` | {float(best['dummy_minus_bos_mean']):.6g} | {float(best['ppl_ratio']):.6g} | {note} |")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    configure_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        done = OUT / cfg.key / 'bias_transfer_sweep.csv'
        if done.exists():
            print(json.dumps({'event':'skip_existing', 'model_key':cfg.key, 'path':str(done)}), flush=True)
            continue
        try:
            run_one(cfg)
        except Exception as exc:
            row = {'model_key': cfg.key, 'family': cfg.family, 'model_id': cfg.model_id, 'error': repr(exc)}
            print(json.dumps(row), flush=True)
            write_csv(OUT / cfg.key / 'bias_transfer_sweep.csv', [row])
        summarize()
    summarize()

if __name__ == '__main__':
    main()
