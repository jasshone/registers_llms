# MMLU Transfer Eval Full Test Summary

Source CSV: `outputs/mmlu_transfer_eval_full_test/summary.csv`

Rows: `5`

| model_key | topk | scale | layers | clean_accuracy | intervened_accuracy | accuracy_delta | dummy_minus_bos_mean | source_wikitext_ppl_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2_medium | 4 | 3 | 3:4 | 0.229241 | 0.228671 | -0.000569719 | 0.328022 | 1.00772 |
| pythia_2_8b | 4 | 3.75 | 2:2 4:2 | 0.246404 | 0.251175 | 0.0047714 | 0.39401 | 1.02118 |
| llama3_2_1b | 8 | 0.87 | 0:2 1:6 | 0.410056 | 0.395813 | -0.014243 | 0.461437 | 1.03752 |
| qwen3_1_7b | 4 | 6 | 2:4 | 0.55042 | 0.549993 | -0.00042729 | 0.537343 | 0.970256 |
| mistral_7b_v0_1 | 1 | 3 | 1:1 | 0.587879 | 0.584176 | -0.00370318 | 0.444973 | 0.990581 |
