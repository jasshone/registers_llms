# Intervention decomposition: llama3_2_3b

Status: complete.

## Command

```bash
.conda-env/bin/python run_intervention_decomposition.py --model-key llama3_2_3b --out outputs/intervention_decomposition/llama3_2_3b_20260714 --seed 17 --max-test-windows 64
```

Script path: `run_intervention_decomposition.py`
Code snapshot: `ab2e4040582a0299e5a69c9fdc816cab1108f988`
Model revision/id: `unsloth/Llama-3.2-3B`
Data split: Wikitext train windows for bias estimation, Wikitext test windows for reported decomposition; windows from `outputs/final_split_bias_transfer/llama3_2_3b/windows_192x1024.pt`.
Seeds: not a training job; random layer-matched control uses fixed seed `17`.
Neuron-list path: selected `outputs/intervention_decomposition/llama3_2_3b_20260714/selected_neurons.pt`, random control `outputs/intervention_decomposition/llama3_2_3b_20260714/layer_matched_random_neurons.pt`.
Output path: `outputs/intervention_decomposition/llama3_2_3b_20260714`.
Accepted relocation row: mask `bos_only`, topk `4`, scale `1.0`, layers `0:1 1:3`.

## Sanity Checks

- Raw per-example rows are written in `per_example.csv`; aggregate condition rows are written in `aggregate_summary.csv`.
- Each condition inserts exactly one zero dummy slot after BOS; only one bias intervention mode is active per condition.
- Random controls are sampled from the same candidate artifact with layer counts matched to the selected neurons.
- Output KL is computed on next-token logits aligned between clean and dummy-inserted sequences.
