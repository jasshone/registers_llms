# Reproducibility Addendum

This bundle now includes rendered figures, exact-number tables, source CSVs, plotting scripts, selected local run/analyze scripts, the `sink_neurons` helper package, git/environment provenance, and fuller result directories for the urgent main-paper figures.

Important limitations still visible in the archive:

- The repository was dirty when captured. See `provenance/git_head.txt`, `provenance/git_status_short.txt`, and `provenance/git_diff_stat.txt`.
- Figure 2 rendered in the urgent set now uses `realtext_toy_shifted_bos_gradients/wikitext103_train_expanded_3seed/aggregate_gradient_summary.csv`, which is the train-text 3-seed shifted-BOS aggregate. For single-position BOS variants, `bos_attention_sum_mean_*` is the attention to the actual BOS position.
- The older cached validation-window position sweep remains preserved under `artifacts/realtext_toy_bos_train_position/wikitext_pythia_cached_val_windows` and as a copied reference source.
- Figure 3 rendered in the urgent set still uses `realtext_toy_bos_presence_controls/wikitext103_train_3seed`, whose aggregate has uneven seed counts. Raw per-seed files and the config are preserved under `artifacts/realtext_toy_bos_presence_controls/wikitext103_train_3seed`.
- Figure 9 provenance is summarized in `provenance/figure9_frozen_selection_provenance.json`. It points to Wikitext-side selection artifacts and the Pile/MMLU evaluation outputs included here.

To regenerate the urgent rendered figures from this archive after extraction, run:

```bash
python code/render_urgent_main_paper_figures.py
```

from the original repository root with the archived files restored to `outputs/`, or adapt the paths in the script to point at the bundled `artifacts/` directories.
