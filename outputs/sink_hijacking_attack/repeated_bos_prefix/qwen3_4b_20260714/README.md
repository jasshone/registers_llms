# Repeated-BOS Prefix Attack: Qwen3-4B

Status: complete.

## Command

```bash
.conda-env/bin/python run_repeated_bos_prefix_attack.py --model-id Qwen/Qwen3-4B --selection outputs/intervention_decomposition/qwen3_4b_20260714/selected_neurons.pt --windows outputs/final_split_bias_transfer_source_abs_bonus0_official_scope/qwen3_4b/windows_192x1024.pt --out outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714 --num-windows 192 --batch-size 1 --counts 8,16,32,64 --scale 4.0 --random-controls 1 --seed 0 --dtype bf16
```

Script path: `run_repeated_bos_prefix_attack.py`

Code snapshot before this artifact commit: `a65525604b51a1e82c512aa342966fdecfcfa76a`

Model revision/id: `Qwen/Qwen3-4B`

Data split/window artifact: `outputs/final_split_bias_transfer_source_abs_bonus0_official_scope/qwen3_4b/windows_192x1024.pt`, 192 windows of length 1024.

Seeds: `0`; layer-matched random-copy control uses deterministic seeds derived from the prefix count and control id.

Neuron-list path: `outputs/intervention_decomposition/qwen3_4b_20260714/selected_neurons.pt` from the frozen Wikitext-selected intervention-decomposition artifacts. This tensor contains the accepted Qwen top-8 selected neurons.

Output path: `outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714`

Log path: `outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714/logs/run.log`

## Protocol

- Prefix counts: `8,16,32,64`.
- Conditions: `bos_repeat_only`, `ordinary_repeat_only`, `selected_copy_bos`, `random_copy_bos`.
- Attack scale: `4.0`, matching the prior Pythia repeated-prefix protocol rather than retuning for Qwen.
- Metrics: inserted-slot attention, original BOS attention, inserted-minus-BOS attention, PPL, and PPL ratio versus clean.

## Sanity Checks

- Clean PPL: `16.72`; clean BOS attention mean: `0.4339`.
- The ordinary-token repeat control stays below clean PPL at all counts; at count 64 its PPL ratio is `0.9668`.
- The BOS-repeat, selected-copy, and layer-matched random-copy conditions move attention onto inserted slots; the PPL effect is much smaller than the Llama run.
- Strongest selected-copy row: count 64, inserted attention `0.5404`, BOS attention `0.0006`, PPL ratio `1.0715`.
- Same-count layer-matched random-copy row: inserted attention `0.5276`, BOS attention `0.0082`, PPL ratio `1.0652`.

Raw rows are in `repeated_bos_prefix_metrics.csv`; the rendered table is in `repeated_bos_prefix_summary.md`.
