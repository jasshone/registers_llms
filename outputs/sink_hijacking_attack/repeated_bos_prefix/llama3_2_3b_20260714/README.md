# Repeated-BOS Prefix Attack: Llama-3.2-3B

Status: complete.

## Command

```bash
.conda-env/bin/python run_repeated_bos_prefix_attack.py --model-id unsloth/Llama-3.2-3B --selection outputs/intervention_decomposition/llama3_2_3b_20260714/selected_neurons.pt --windows outputs/final_split_bias_transfer_source_abs_bonus0_official_scope/llama3_2_3b/windows_192x1024.pt --out outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714 --num-windows 192 --batch-size 1 --counts 8,16,32,64 --scale 4.0 --random-controls 1 --seed 0 --dtype bf16
```

Script path: `run_repeated_bos_prefix_attack.py`

Code snapshot before this artifact commit: `ed679de72f82fb6d2792da2e36dc6e6e352513fa`

Model revision/id: `unsloth/Llama-3.2-3B`

Data split/window artifact: `outputs/final_split_bias_transfer_source_abs_bonus0_official_scope/llama3_2_3b/windows_192x1024.pt`, 192 windows of length 1024.

Seeds: `0`; layer-matched random-copy control uses deterministic seeds derived from the prefix count and control id.

Neuron-list path: `outputs/intervention_decomposition/llama3_2_3b_20260714/selected_neurons.pt` from the frozen Wikitext-selected intervention-decomposition artifacts. This tensor contains the accepted Llama top-4 selected neurons.

Output path: `outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714`

Log path: `outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714/logs/run.log`

## Protocol

- Prefix counts: `8,16,32,64`.
- Conditions: `bos_repeat_only`, `ordinary_repeat_only`, `selected_copy_bos`, `random_copy_bos`.
- Attack scale: `4.0`, matching the prior Pythia repeated-prefix protocol rather than retuning for Llama.
- Metrics: inserted-slot attention, original BOS attention, inserted-minus-BOS attention, PPL, and PPL ratio versus clean.

## Sanity Checks

- Clean PPL: `8.963`; clean BOS attention mean: `0.6274`.
- The ordinary-token repeat control stays near clean PPL at all counts; at count 64 its PPL ratio is `0.9963`.
- The BOS-repeat and selected-copy conditions both move attention onto inserted slots and sharply increase PPL with prefix count.
- Strongest selected-copy row: count 64, inserted attention `0.7528`, BOS attention `0.0012`, PPL ratio `21.3277`.
- Same-count layer-matched random-copy row: inserted attention `0.7371`, BOS attention `0.0115`, PPL ratio `19.1000`.

Raw rows are in `repeated_bos_prefix_metrics.csv`; the rendered table is in `repeated_bos_prefix_summary.md`.
