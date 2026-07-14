# Full Wikitext Causal No-Op Run

Status: completed for Wikitext-test with full available split coverage for Llama and Qwen after the initial 320-window run. Pythia, Llama, and Qwen all have primary site-level rows, sequence-level paired summaries, bootstrap summaries, value-content validation, and secondary all-head aggregate outputs.

Code snapshot before this artifact commit: `caf82d70c5544a352337fa74ec92b1968d88b867`.

Scripts:

- `run_causal_noop_experiment.py`
- `sink_neurons/attention_noop.py`
- `audit_causal_noop_outputs.py`
- `verify_causal_noop_experiment.py`

Models:

- `pythia_1b`: `EleutherAI/pythia-1b`
- `llama3_2_3b`: `unsloth/Llama-3.2-3B`
- `qwen3_4b`: `Qwen/Qwen3-4B`

Data and selection:

- Head-selection split: Wikitext-103 train, 256 windows, length 256, stride 256.
- Evaluation split: Wikitext-103 test, length 256, stride 256.
- Fixed sampling seed: `17`.
- Primary delta: `0.05`.
- Sensitivity deltas: `0.02`, `0.10`.
- Primary design: one sampled eligible query, one sink-heavy head, and one layer-matched low-sink head per sequence; paired sink, early ordinary, and random ordinary targets are averaged at sequence level.
- Each replicated batch element receives exactly one intervention via batched per-example intervention specifications.

Primary commands:

```bash
.conda-env/bin/python run_causal_noop_experiment.py --models pythia_1b llama3_2_3b qwen3_4b --out outputs/causal_noop_full_wikitext_20260714 --head-train-windows 256 --eval-windows 320 --sensitivity-windows 128 --value-windows 128 --secondary-windows 128 --window-length 256 --stride 256 --batch-size 1 --site-batch-size 12 --query-start 16 --heads 16 --primary-delta 0.05 --sensitivity-deltas 0.02 0.10 --datasets wikitext_test --seed 17 --bootstrap-samples 2000 --flush-every 24
```

Coverage extensions:

```bash
.conda-env/bin/python run_causal_noop_experiment.py --models llama3_2_3b --out outputs/causal_noop_full_wikitext_20260714 --head-train-windows 256 --eval-start 320 --eval-windows 813 --sensitivity-windows 813 --value-windows 128 --secondary-windows 128 --window-length 256 --stride 256 --batch-size 1 --site-batch-size 12 --query-start 16 --heads 16 --primary-delta 0.05 --sensitivity-deltas 0.02 0.10 --datasets wikitext_test --seed 17 --bootstrap-samples 2000 --flush-every 24 --skip-value-validation --skip-secondary --no-progress
```

```bash
.conda-env/bin/python run_causal_noop_experiment.py --models qwen3_4b --out outputs/causal_noop_full_wikitext_20260714 --head-train-windows 256 --eval-start 320 --eval-windows 852 --sensitivity-windows 852 --value-windows 128 --secondary-windows 128 --window-length 256 --stride 256 --batch-size 1 --site-batch-size 12 --query-start 16 --heads 16 --primary-delta 0.05 --sensitivity-deltas 0.02 0.10 --datasets wikitext_test --seed 17 --bootstrap-samples 2000 --flush-every 24 --skip-value-validation --skip-secondary --no-progress
```

Report refresh:

```bash
.conda-env/bin/python run_causal_noop_experiment.py --models llama3_2_3b qwen3_4b --out outputs/causal_noop_full_wikitext_20260714 --head-train-windows 256 --eval-windows 320 --sensitivity-windows 128 --value-windows 128 --secondary-windows 128 --window-length 256 --stride 256 --batch-size 1 --site-batch-size 12 --query-start 16 --heads 16 --primary-delta 0.05 --sensitivity-deltas 0.02 0.10 --datasets wikitext_test --seed 17 --bootstrap-samples 2000 --flush-every 24 --no-progress
```

Key outputs:

- Top-level report: `causal_noop_experiment.md`
- Machine-readable summary: `run_summary.json`
- Per-model head selections: `<model>/head_selection.json`
- Per-model primary raw rows: `<model>/wikitext_test/site_results.csv`
- Per-model sequence paired rows: `<model>/wikitext_test/sequence_summaries.csv`
- Per-model primary bootstraps: `<model>/wikitext_test/bootstrap_summaries.csv`
- Per-model value validation: `<model>/wikitext_test/value_content_results.csv`, `value_sequence_summaries.csv`, `value_bootstrap_summaries.csv`
- Per-model secondary all-head aggregate: `<model>/wikitext_test/secondary_results.csv`, `secondary_sequence_summaries.csv`, `secondary_bootstrap_summaries.csv`
- Logs: `logs/`

Sanity checks:

- Implementation verifier command:

```bash
.conda-env/bin/python verify_causal_noop_experiment.py
```

Log: `logs/verify_causal_noop_experiment.log`. Result: passed for Llama-style, GPT-NeoX-style, and Qwen3-style attention modules.

- Strict Pythia/Llama audit command:

```bash
.conda-env/bin/python audit_causal_noop_outputs.py --out outputs/causal_noop_full_wikitext_20260714 --models pythia_1b llama3_2_3b --datasets wikitext_test --deltas 0.02 0.05 0.10 --min-sequences 256 --primary-delta 0.05 --sensitivity-min-sequences 64 --value-min-sequences 128 --secondary-min-sequences 128
```

Log: `logs/audit_strict_pythia_llama.log`. Result: passed, with a warning that Llama has no middle-layer primary rows because the selected heads fall in early/late layer bands.

- Qwen full-available Wikitext audit command:

```bash
.conda-env/bin/python audit_causal_noop_outputs.py --out outputs/causal_noop_full_wikitext_20260714 --models qwen3_4b --datasets wikitext_test --deltas 0.02 0.05 0.10 --min-sequences 180 --primary-delta 0.05 --sensitivity-min-sequences 63 --value-min-sequences 128 --secondary-min-sequences 128
```

Log: `logs/audit_qwen_full_available_wikitext.log`. Result: passed.

Coverage caveat:

- The strict all-model audit with `--min-sequences 256 --sensitivity-min-sequences 64` fails only for Qwen sink-heavy pairs after using all tokenized Wikitext-test windows. Qwen has 1,172 available Wikitext-test windows at length 256/stride 256; after probability-boundary skips, sink-heavy all-layer coverage is 187 paired sequences at `delta=0.05` and 63 paired sequences at `delta=0.10`.
- Pythia and Llama pass the strict 256 primary / 64 sensitivity thresholds. Llama has 258 sink-heavy all-layer paired sequences at `delta=0.05`; Pythia has 273.

Key primary sink-heavy all-layer results at `delta=0.05`:

| Model | Control | Mean control-minus-sink KL | 95% bootstrap CI | Sequences |
| --- | --- | ---: | --- | ---: |
| Pythia-1B | early ordinary | 0.000148 | [0.000075, 0.000230] | 273 |
| Pythia-1B | random ordinary | 0.000165 | [0.000099, 0.000239] | 273 |
| Llama-3.2-3B | early ordinary | -0.000005 | [-0.000085, 0.000077] | 258 |
| Llama-3.2-3B | random ordinary | -0.000073 | [-0.000145, 0.000001] | 258 |
| Qwen3-4B | early ordinary | 0.000027 | [-0.000097, 0.000145] | 187 |
| Qwen3-4B | random ordinary | 0.000005 | [-0.000119, 0.000124] | 187 |

Value validation all-layer results:

| Model | Mean ordinary-at-sink minus sink-at-ordinary KL | 95% bootstrap CI | Sequences |
| --- | ---: | --- | ---: |
| Pythia-1B | 0.083782 | [0.062268, 0.106484] | 128 |
| Llama-3.2-3B | 0.671896 | [0.421045, 0.950387] | 128 |
| Qwen3-4B | 0.026213 | [0.017103, 0.037786] | 128 |
