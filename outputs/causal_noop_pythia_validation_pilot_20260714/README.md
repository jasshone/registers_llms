# Pythia causal No-Op validation pilot

Status: complete, with validation caveat. The run evaluated 128 Wikitext-test windows at delta=0.05, but the strict audit requiring 128 non-skipped sink-heavy paired sequences failed because 22 sink-heavy sink-target sites were skipped when target attention already exceeded the valid +delta range. Do not use this as the expansion gate without a follow-up coverage pilot or eligibility prefilter.

## Command

```bash
.conda-env/bin/python run_causal_noop_experiment.py --models pythia_1b --out outputs/causal_noop_pythia_validation_pilot_20260714 --head-train-windows 256 --eval-windows 128 --window-length 256 --stride 256 --batch-size 1 --site-batch-size 12 --query-start 16 --heads 16 --primary-delta 0.05 --deltas 0.05 --value-windows 128 --secondary-windows 128 --datasets wikitext_test --seed 17 --bootstrap-samples 2000 --flush-every 24
```

Script path: `run_causal_noop_experiment.py`
Verifier run first: `.conda-env/bin/python verify_causal_noop_experiment.py` passed.
Code snapshot at run start: current branch after commit `f313866` plus existing dirty worktree; this output is committed after verification.
Model revision/id: `EleutherAI/pythia-1b`
Data split: Wikitext train for head selection, Wikitext test for pilot evaluation.
Seeds: fixed random seed `17`.
Head-selection path: `pythia_1b/head_selection.json`.
Output path: `outputs/causal_noop_pythia_validation_pilot_20260714`.

## Output Checks

- Primary site rows: `768` rows = 128 sequences x 2 head kinds x 3 targets x 1 delta.
- Value validation rows: `256` rows = 128 sequences x 2 swap directions.
- Secondary all-head rows: `384` rows = 128 sequences x 3 targets.
- Main sink-heavy all-layer paired KL differences were positive for early and random controls, but used `106` non-skipped sequences after validity filtering.
- Strict audit command below failed only on sink-heavy primary valid-sequence coverage:

```bash
.conda-env/bin/python audit_causal_noop_outputs.py --out outputs/causal_noop_pythia_validation_pilot_20260714 --models pythia_1b --datasets wikitext_test --deltas 0.05 --min-sequences 128 --primary-delta 0.05 --value-min-sequences 128 --secondary-min-sequences 128
```

Failure:

```text
[error] pythia_1b/wikitext_test: primary sink_heavy/early/delta=0.05 has < 128 sequences
[error] pythia_1b/wikitext_test: primary sink_heavy/random/delta=0.05 has < 128 sequences
```

Recommended next step: rerun a coverage pilot with more eval windows or add a query eligibility prefilter so the expansion gate has at least 128 non-skipped paired sink-heavy sequences.
