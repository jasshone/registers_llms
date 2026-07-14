# Intervention decomposition: mistral_7b_v0_1

Status: complete.

## Command

```bash
HF_HUB_DISABLE_XET=1 .conda-env/bin/python run_intervention_decomposition.py --model-key mistral_7b_v0_1 --root outputs/final_split_bias_transfer --out outputs/intervention_decomposition/mistral_7b_v0_1_20260714_run3 --seed 17 --max-test-windows 64
```

Script path: `run_intervention_decomposition.py`
Code snapshot: `9acf131c3b5bd8479925ef451fc534f502c640e1`
Model revision/id: `mistralai/Mistral-7B-v0.1`
Data split: Wikitext train windows for bias estimation, Wikitext test windows for reported decomposition; windows from `outputs/final_split_bias_transfer/mistral_7b_v0_1/windows_192x1024.pt`.
Seeds: not a training job; random layer-matched control uses fixed seed `17`.
Neuron-list path: selected `outputs/intervention_decomposition/mistral_7b_v0_1_20260714_run3/selected_neurons.pt`, random control `outputs/intervention_decomposition/mistral_7b_v0_1_20260714_run3/layer_matched_random_neurons.pt`.
Output path: `outputs/intervention_decomposition/mistral_7b_v0_1_20260714_run3`.
Accepted relocation row: mask `bos_only`, topk `1`, scale `3.0`, layers `1:1`.

## Sanity Checks

- Raw per-example rows are written in `per_example.csv`; aggregate condition rows are written in `aggregate_summary.csv`.
- Each condition inserts exactly one zero dummy slot after BOS; only one bias intervention mode is active per condition.
- Random controls are sampled from the same candidate artifact with layer counts matched to the selected neurons.
- Output KL is computed on next-token logits aligned between clean and dummy-inserted sequences.
