# MMLU Transfer Eval Full Test Summary

Source CSV: `outputs/mmlu_transfer_eval_full_test/summary.csv`

Rows: `7`

| model_key | model_id | status | eval_examples | mask | topk | scale | layers | clean_accuracy | intervened_accuracy | accuracy_delta | dummy_minus_bos_mean | source_wikitext_ppl_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2_medium | gpt2-medium | ok | 14042 | bos_only | 4 | 3 | 3:4 | 0.229241 | 0.228671 | -0.000569719 | 0.328022 | 1.00772 |
| opt_350m | facebook/opt-350m | ok | 14042 | bos_only | 8 | 1 | 0:3 1:4 2:1 | 0.227318 | 0.22625 | -0.00106822 | -0.07934 | 1.03864 |
| pythia_2_8b | EleutherAI/pythia-2.8b | ok | 14042 | bos_only | 4 | 3.75 | 2:2 4:2 | 0.246404 | 0.251175 | 0.0047714 | 0.39401 | 1.02118 |
| phi_2 | microsoft/phi-2 | ok | 14042 | bos_only | 16 | 2 | 1:6 2:10 | 0.513887 | 0.52521 | 0.0113232 | 0.605105 | 1.04773 |
| llama3_2_1b | unsloth/Llama-3.2-1B | ok | 14042 | bos_only | 8 | 0.87 | 0:2 1:6 | 0.410056 | 0.395813 | -0.014243 | 0.461437 | 1.03752 |
| qwen3_1_7b | Qwen/Qwen3-1.7B | ok | 14042 | bos_only | 4 | 6 | 2:4 | 0.55042 | 0.549993 | -0.00042729 | 0.537343 | 0.970256 |
| mistral_7b_v0_1 | mistralai/Mistral-7B-v0.1 | ok | 14042 | bos_only | 1 | 3 | 1:1 | 0.587879 | 0.584176 | -0.00370318 | 0.444973 | 0.990581 |
