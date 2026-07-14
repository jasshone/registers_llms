# Preserved Main-Paper Figures

Rendered from existing result files; no training or exhaustive grid was run.

## Figures
- Figure 2: `figure2_bos_position_sweep.{png,pdf}`
- Figure 3: `figure3_bos_identity_controls.{png,pdf}`
- Figure 7: `figure7_toy_pretrained_bridge.{png,pdf}`
- Figure 8: `figure8_repeated_bos_prefix_attack.{png,pdf}`
- Figure 9: `figure9_frozen_selection_generalization.{png,pdf}`

## Exact Number Tables
- Figure 2: `tables/figure2_bos_position_sweep_exact.{csv,md}`
- Figure 3: `tables/figure3_bos_identity_controls_exact.{csv,md}`
- Figure 7: `tables/figure7_toy_pretrained_bridge_exact.{csv,md}`
- Figure 8: `tables/figure8_repeated_bos_prefix_attack_exact.{csv,md}`
- Figure 9 Wikitext: `tables/figure9_wikitext_exact.{csv,md}`
- Figure 9 Pile: `tables/figure9_pile_exact.{csv,md}`
- Figure 9 MMLU: `tables/figure9_mmlu_exact.{csv,md}`

## Source CSVs
- figure2: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/realtext_toy_shifted_bos_gradients/wikitext103_train_expanded_3seed/aggregate_gradient_summary.csv`
- figure2_cached_reference: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/realtext_toy_bos_train_position/wikitext_pythia_cached_val_windows/bos_train_position_summary.csv`
- figure3: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/realtext_toy_bos_presence_controls/wikitext103_train_3seed/aggregate_summary.csv`
- figure7: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/toy_pretrained_bridge/wikitext_pythia_tokenizer_10k_scale10_test512/toy_relocation_bridge_metrics.csv`
- figure8: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/sink_hijacking_attack/repeated_bos_prefix/pythia_1b_full/repeated_bos_prefix_metrics.csv`
- figure9_wikitext: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/final_split_bias_transfer_source_abs_bonus0/final_results.csv`
- figure9_pile: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/pile_val_sharded_transfer_full/summary_shards.csv`
- figure9_mmlu: `/workspace/registers_llms/outputs/preserved_main_paper_figures/artifacts/mmlu_transfer_eval_full_test/summary.csv`

Notes:
- Figure 2 now uses the train-text Wikitext103 expanded 3-seed shifted-BOS aggregate. The older cached validation-window sweep is copied as `figure2_cached_reference_bos_train_position_summary.csv`.
- Figure 3 uses the available aggregate seed summary; variants with a single seed have zero/blank visible error bars.
