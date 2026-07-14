# Bootstrap Postprocessing Summary

## Intervention Decomposition

| model | condition | group | n | dummy-BOS mean | 95% CI | PPL ratio | 95% CI |
|---|---|---|---:|---:|---|---:|---|
| pythia_1b | dummy_add_only | selected | 64 | 0.0103226 | [0.00985598, 0.0107872] | 1.00784 | [1.00616, 1.00953] |
| pythia_1b | dummy_only | selected | 64 | -0.348284 | [-0.35347, -0.343167] | 0.996428 | [0.994856, 0.997896] |
| pythia_1b | full_transfer | selected | 64 | 0.313848 | [0.308991, 0.318643] | 1.0072 | [1.00563, 1.00875] |
| pythia_1b | random_dummy_add_only | layer_matched_random | 64 | -0.227572 | [-0.231249, -0.22378] | 0.994829 | [0.993327, 0.996284] |
| pythia_1b | random_full_transfer | layer_matched_random | 64 | -0.229975 | [-0.233811, -0.226138] | 0.997057 | [0.99544, 0.998636] |
| pythia_1b | source_subtract_only | selected | 64 | -0.159189 | [-0.163248, -0.155409] | 1.43428 | [1.41148, 1.45805] |
| llama3_2_3b | dummy_add_only | selected | 64 | -0.0733503 | [-0.0737328, -0.072973] | 1.01837 | [1.01601, 1.02076] |
| llama3_2_3b | dummy_only | selected | 64 | -0.626408 | [-0.630331, -0.622594] | 0.99022 | [0.98877, 0.991653] |
| llama3_2_3b | full_transfer | selected | 64 | 0.529108 | [0.525221, 0.53301] | 1.03526 | [1.03067, 1.03986] |
| llama3_2_3b | random_dummy_add_only | layer_matched_random | 64 | -0.626424 | [-0.630346, -0.622511] | 0.99013 | [0.988617, 0.991619] |
| llama3_2_3b | random_full_transfer | layer_matched_random | 64 | -0.626516 | [-0.630448, -0.622551] | 0.989846 | [0.988302, 0.991361] |
| llama3_2_3b | source_subtract_only | selected | 64 | -0.0462639 | [-0.0464598, -0.0460795] | 20.1929 | [18.61, 21.8679] |
| qwen3_4b | dummy_add_only | selected | 64 | 0.00022472 | [0.00021208, 0.000237961] | 0.959981 | [0.954567, 0.965009] |
| qwen3_4b | dummy_only | selected | 64 | 0.000178395 | [0.000164266, 0.000192454] | 0.960512 | [0.955256, 0.965405] |
| qwen3_4b | full_transfer | selected | 64 | 0.419753 | [0.416327, 0.423292] | 0.965831 | [0.96008, 0.97159] |
| qwen3_4b | random_dummy_add_only | layer_matched_random | 64 | 0.000179938 | [0.000166779, 0.000193643] | 0.960052 | [0.954816, 0.964934] |
| qwen3_4b | random_full_transfer | layer_matched_random | 64 | 0.000206466 | [0.000192442, 0.000220943] | 0.960207 | [0.954803, 0.965306] |
| qwen3_4b | source_subtract_only | selected | 64 | 0.419123 | [0.415852, 0.422553] | 0.967925 | [0.962316, 0.97337] |

## Toy Bridge Random Controls

| step | n random controls | random dummy-BOS mean | 95% CI | selected dummy-BOS | selected-random |
|---:|---:|---:|---|---:|---:|
| 0 | 5 | 3.9873e-05 | [-0.000117726, 0.000203568] | -0.000180163 | -0.000220036 |
| 1000 | 5 | 0.0157657 | [0.0054035, 0.026128] | 0.0314649 | 0.0156991 |
| 2000 | 5 | 0.0235779 | [0.00922706, 0.0334015] | 0.0429979 | 0.01942 |
| 3000 | 5 | -0.0108429 | [-0.0180935, -0.00371548] | 0.0530394 | 0.0638823 |
| 4000 | 5 | -0.0142968 | [-0.0244206, -0.00346589] | 0.0454047 | 0.0597015 |
| 5000 | 5 | -0.00847125 | [-0.0205053, 0.00356278] | 0.0237289 | 0.0322002 |
| 6000 | 5 | -0.0235048 | [-0.034721, -0.0122885] | 0.0238741 | 0.0473789 |
| 7000 | 5 | -0.0217812 | [-0.035555, -0.00579163] | 0.0160497 | 0.037831 |
| 8000 | 5 | -0.0384455 | [-0.0424303, -0.0345488] | 0.0263457 | 0.0647912 |
| 9000 | 5 | -0.0465849 | [-0.053423, -0.0417916] | 0.0156887 | 0.0622736 |
| 10000 | 5 | -0.0338428 | [-0.0433381, -0.021534] | 0.0148649 | 0.0487078 |

## Aggregate-Only Sources

| artifact | status | rows | source |
|---|---|---:|---|
| pile | aggregate_only_no_bootstrap | 15 | `outputs/preserved_main_paper_figures/sources/figure9_pile_summary_shards.csv` |
| mmlu | aggregate_only_no_bootstrap | 7 | `outputs/preserved_main_paper_figures/sources/figure9_mmlu_summary.csv` |
| repeated_bos_pythia_1b | aggregate_only_no_bootstrap | 29 | `outputs/preserved_main_paper_figures/sources/figure8_repeated_bos_prefix_metrics.csv` |
| repeated_bos_llama3_2_3b | aggregate_only_no_bootstrap | 17 | `outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714/repeated_bos_prefix_metrics.csv` |
| repeated_bos_qwen3_4b | aggregate_only_no_bootstrap | 17 | `outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714/repeated_bos_prefix_metrics.csv` |
