# Bootstrap Postprocessing

Status: complete for preserved artifacts that contain a valid CPU-only resampling unit.

Command:

```bash
.conda-env/bin/python postprocess_bootstrap_artifacts.py --out outputs/bootstrap_postprocess_20260714 --seed 17 --samples 10000
```

Code snapshot before this artifact commit: `a148f18cd74189c3ad54c3943027295524248ba2`

Outputs:

- `intervention_decomposition_bootstrap.csv`: nonparametric example-level bootstrap CIs for Pythia-1B, Llama-3.2-3B, and Qwen3-4B intervention-decomposition per-example rows.
- `toy_bridge_random_control_bootstrap.csv`: random-control-run bootstrap CIs for Figure 7 toy bridge random controls.
- `aggregate_only_status.csv`: explicit status for Pile, MMLU, and repeated-prefix attack sources where only aggregate rows are preserved.
- `summary.md`: compact human-readable summary.

Sanity-check status:

- Intervention-decomposition bootstrap rows with example-level support: `18`.
- Toy-bridge random-control bootstrap rows: `11`.
- Aggregate-only source statuses: `5`.

Important limitation:

Pile, MMLU, and repeated-prefix attack artifacts in the preserved main-paper bundle contain aggregate rows only. This script deliberately does not invent confidence intervals from aggregate means, single shards, or reported sample counts.
