# Job 1: Completed BOS Identity-Control Seeds

Status: complete. All six identity-control variants now have final-step rows for seeds 0, 1, and 2 in `final_rows_by_seed.csv`; aggregate means/stds are in `aggregate_summary.csv`.

Exact commands run for new completions:

```bash
.conda-env/bin/python run_realtext_toy_bos_presence_traintext_3seed.py --out outputs/realtext_toy_bos_presence_controls/wikitext103_train_missing_seed_completion_20260714 --variants no_bos single_8 single_32 duplicate_0,8 duplicate_0,32 --seeds 1 2 --token-cache outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt --steps 10000 --batch-size 64 --eval-batch-size 128 --eval-interval 1000 --eval-split val --seq-len 128 --n-layer 4 --n-head 4 --d-model 128 --d-mlp 512 --lr 0.0003 --weight-decay 0.1 --device cuda
```

This first command was interrupted after completing seed 1 for all listed variants and seed 2 for `no_bos`; partial `single_8` seed 2 files are preserved but not used in the completed aggregate.

```bash
.conda-env/bin/python run_realtext_toy_bos_presence_traintext_3seed.py --out outputs/realtext_toy_bos_presence_controls/wikitext103_train_missing_seed_completion_20260714_resume2 --variants single_8 single_32 duplicate_0,8 duplicate_0,32 --seeds 2 --token-cache outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt --steps 10000 --batch-size 64 --eval-batch-size 128 --eval-interval 1000 --eval-split val --seq-len 128 --n-layer 4 --n-head 4 --d-model 128 --d-mlp 512 --lr 0.0003 --weight-decay 0.1 --device cuda
```

Additional completion for the remaining `single_0` seed 2, so every plotted condition has three seeds:

```bash
.conda-env/bin/python run_realtext_toy_bos_presence_traintext_3seed.py --out outputs/realtext_toy_bos_presence_controls/wikitext103_train_missing_seed_completion_20260714_single0_seed2 --variants single_0 --seeds 2 --token-cache outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt --steps 10000 --batch-size 64 --eval-batch-size 128 --eval-interval 1000 --eval-split val --seq-len 128 --n-layer 4 --n-head 4 --d-model 128 --d-mlp 512 --lr 0.0003 --weight-decay 0.1 --device cuda
```

Script path: `run_realtext_toy_bos_presence_traintext_3seed.py` using `run_realtext_toy_bos_presence_controls.py` and `run_realtext_toy_sink_training.py`.

Code snapshot/commit: `443bacfa57c12b70062df5429b951bca3a4d4e6b`. Worktree has unrelated dirty files; this job reused existing scripts without editing them.

Model revision: toy `RealTextTinyGPT` config copied in `original_config.json`; tokenizer source from token cache metadata is `EleutherAI/pythia-70m`.

Data split: `Salesforce/wikitext` / `wikitext-103-raw-v1` / train token cache `outputs/realtext_toy_data/wikitext103_train10m_pythia70m_tokens.pt`; eval split `val`.

Seeds: 0, 1, 2 for `single_0`, `no_bos`, `single_8`, `single_32`, `duplicate_0,8`, `duplicate_0,32`.

Neuron-list path: not applicable; this is a toy BOS identity-control training job.

Output path: `outputs/realtext_toy_bos_presence_controls/wikitext103_train_completed_3seed_20260714`. Raw per-step metrics are preserved in the source directories listed by `source_manifest.json`; logs are in each source directory under `logs/run.log` when generated.

Sanity-check status: passed basic output checks. The merged table has exactly one final row for each variant/seed pair, all selected rows are at step 10000, and all six variants have `n_seeds=3` in `aggregate_summary.csv`.
