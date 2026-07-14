# Pile Validation Sharded Transfer Summary

Source CSV: `outputs/pile_val_sharded_transfer_full/summary_shards.csv`

Rows: `15`

| model_key | model_id | status | selected_windows_seen | total_windows_seen | mask | topk | scale | candidate_group | layers | dummy_minus_bos_mean | baseline_ppl | intervened_ppl | ppl_ratio | source_wikitext_ppl_ratio | eval_windows_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | gpt2 | ok | 374864 | 374864 | bos_only | 4 | 3 |  | 1:1 2:3 | 0.313449 | 19.5717 | 19.6748 | 1.00527 | 1.00228 | 60.1947 |
| gpt2_medium | gpt2-medium | ok | 374864 | 374864 | bos_only | 4 | 3 |  | 3:4 | 0.318555 | 14.4465 | 14.5538 | 1.00743 | 1.00772 | 29.2697 |
| llama3_2_1b | unsloth/Llama-3.2-1B | ok | 320055 | 320055 | bos_only | 8 | 0.87 | pipeline_sinks_plus_bos | 0:2 1:6 | 0.404859 | 9.33002 | 9.67992 | 1.0375 | 1.03752 | 14.8063 |
| mistral_7b_v0_1 | mistralai/Mistral-7B-v0.1 | ok | 376769 | 376769 | bos_only | 1 | 3 |  | 1:1 | 0.335661 | 5.41674 | 5.39311 | 0.995637 | 0.990581 | 4.58163 |
| opt_125m | facebook/opt-125m | ok | 374498 | 374498 | bos_only | 8 | 3 |  | 0:8 | 0.461928 | 15.9151 | 16.0273 | 1.00705 | 1.00166 | 60.2571 |
| opt_350m | facebook/opt-350m | ok | 374498 | 374498 | bos_only | 8 | 1 |  | 0:3 1:4 2:1 | 0.0151405 | 13.2529 | 13.4824 | 1.01731 | 1.03864 | 28.5467 |
| phi_1_5 | microsoft/phi-1_5 | ok | 348783 | 348783 | bos_only | 2 | 3 |  | 0:1 1:1 | 0.40717 | 13.6789 | 13.6824 | 1.00026 | 1.00274 | 11.8686 |
| phi_2 | microsoft/phi-2 | ok | 348783 | 348783 | bos_only | 16 | 2 |  | 1:6 2:10 | 0.51687 | 9.19255 | 9.3734 | 1.01967 | 1.04773 | 7.56475 |
| pythia_160m | EleutherAI/pythia-160m | ok | 333072 | 333072 | bos_only | 4 | 3 |  | 1:1 3:3 | 0.112668 | 33.725 | 34.3018 | 1.0171 | 1.00671 | 59.0804 |
| pythia_1_4b | EleutherAI/pythia-1.4b | ok | 333072 | 333072 | bos_only | 4 | 3 |  | 3:4 | 0.374317 | 8.58942 | 8.61458 | 1.00293 | 1.00389 | 17.4673 |
| pythia_1b | EleutherAI/pythia-1b | ok | 333072 | 333072 | bos_only | 16 | 2 |  | 3:14 4:2 | 0.3423 | 8.92141 | 8.92506 | 1.00041 | 1.00855 | 27.0753 |
| pythia_2_8b | EleutherAI/pythia-2.8b | ok | 333072 | 333072 | bos_only | 4 | 3.75 | early_0_8 | 2:2 4:2 | 0.218776 | 7.62646 | 7.67718 | 1.00665 | 1.02118 | 8.36198 |
| pythia_410m | EleutherAI/pythia-410m | ok | 333072 | 333072 | bos_only | 8 | 2.25 |  | 5:5 6:3 | 0.33407 | 11.4318 | 11.5928 | 1.01408 | 1.01776 | 27.3551 |
| pythia_70m | EleutherAI/pythia-70m | ok | 333072 | 333072 | bos_only | 2 | 3 |  | 2:2 | 0.0058 | 140.693 | 142.449 | 1.01248 | 1.00917 | 104.5 |
| qwen3_1_7b | Qwen/Qwen3-1.7B | ok | 332809 | 332809 | bos_only | 4 | 6 |  | 2:4 | 0.455657 | 12.2497 | 12.0757 | 0.985797 | 0.970256 | 11.5796 |
