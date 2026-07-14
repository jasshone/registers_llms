# Pythia causal No-Op primary coverage pilot

Status: complete. This is a primary-only follow-up to `outputs/causal_noop_pythia_validation_pilot_20260714`, run with 192 Wikitext-test windows to obtain at least 128 non-skipped sink-heavy paired sequences at delta=0.05. It reuses the frozen head selection from the 128-window pilot.

## Command

```bash
.conda-env/bin/python run_causal_noop_experiment.py --models pythia_1b --out outputs/causal_noop_pythia_validation_pilot_coverage_20260714 --head-train-windows 256 --eval-windows 192 --window-length 256 --stride 256 --batch-size 1 --site-batch-size 12 --query-start 16 --heads 16 --primary-delta 0.05 --deltas 0.05 --value-windows 0 --secondary-windows 0 --datasets wikitext_test --seed 17 --bootstrap-samples 2000 --flush-every 24 --skip-value-validation --skip-secondary
```

Setup command copied the frozen head selection and train-window cache from the 128-window pilot before running:

```bash
mkdir -p outputs/causal_noop_pythia_validation_pilot_coverage_20260714/pythia_1b/windows outputs/causal_noop_pythia_validation_pilot_coverage_20260714/logs && cp outputs/causal_noop_pythia_validation_pilot_20260714/pythia_1b/head_selection.json outputs/causal_noop_pythia_validation_pilot_coverage_20260714/pythia_1b/head_selection.json && cp outputs/causal_noop_pythia_validation_pilot_20260714/pythia_1b/windows/wikitext_train_256x256_s256.pt outputs/causal_noop_pythia_validation_pilot_coverage_20260714/pythia_1b/windows/wikitext_train_256x256_s256.pt
```

Script path: `run_causal_noop_experiment.py`
Model revision/id: `EleutherAI/pythia-1b`
Data split: Wikitext train for frozen head selection, Wikitext test for evaluation.
Seeds: fixed random seed `17`.
Head-selection path: `pythia_1b/head_selection.json` copied from the 128-window pilot.
Output path: `outputs/causal_noop_pythia_validation_pilot_coverage_20260714`.

## Primary Checks

- Primary site rows: `1152` rows = 192 sequences x 2 head kinds x 3 targets x 1 delta.
- Valid all-layer paired sequence counts: low_sink early/random `191`, sink_heavy early/random `164`.
- Sink-heavy all-layer early control: mean control-sink KL `0.000156888`, 95% CI `[0.0000536554, 0.00027386]`, sink less disruptive fraction `0.634`.
- Sink-heavy all-layer random control: mean control-sink KL `0.000148348`, 95% CI `[0.0000542972, 0.000248444]`, sink less disruptive fraction `0.671`.
- The generic `audit_causal_noop_outputs.py` is not used for this primary-only artifact because it requires value and secondary files; those are supplied by the 128-window pilot artifact.
