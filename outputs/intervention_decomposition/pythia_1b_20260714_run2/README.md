# Intervention decomposition: pythia_1b

Status: complete.

## Command

```bash
.conda-env/bin/python run_intervention_decomposition.py --model-key pythia_1b --out outputs/intervention_decomposition/pythia_1b_20260714_run2 --seed 17 --max-test-windows 64
```

Script path: `run_intervention_decomposition.py`
Code snapshot at run start: `b46acdc0321edfbcfd005a727ed1887fd82c5289`; runner and artifacts are committed together after verification.
Model revision/id: `EleutherAI/pythia-1b`
Data split: Wikitext train windows for bias estimation, Wikitext test windows for reported decomposition; windows from `outputs/final_split_bias_transfer/pythia_1b/windows_192x1024.pt`.
Seeds: not a training job; random layer-matched control uses fixed seed `17`.
Neuron-list path: selected `outputs/intervention_decomposition/pythia_1b_20260714_run2/selected_neurons.pt`, random control `outputs/intervention_decomposition/pythia_1b_20260714_run2/layer_matched_random_neurons.pt`.
Output path: `outputs/intervention_decomposition/pythia_1b_20260714_run2`.
Accepted relocation row: mask `pipeline_sinks_plus_bos`, topk `16`, scale `2.0`, layers `3:14 4:2`.

## Sanity Checks

- Raw per-example rows: `384` rows in `per_example.csv`; aggregate condition rows: `6` rows in `aggregate_summary.csv`.
- Each condition inserts exactly one zero dummy slot after BOS; only one bias intervention mode is active per condition.
- Random controls are sampled from the same candidate artifact with layer counts matched to the selected neurons.
- Output KL is computed on next-token logits aligned between clean and dummy-inserted sequences.
