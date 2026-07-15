# Repeated-BOS Amplification Statistics

This postprocesses existing repeated-BOS aggregate outputs. The paired unit is prefix count within each model, not individual windows, because the repeated-BOS runner preserved only aggregate metrics. Positive differences mean `selected_copy_bos` is larger than the named control at the same prefix count.

Sources:
- `pythia_1b`: `outputs/preserved_main_paper_figures/sources/figure8_repeated_bos_prefix_metrics.csv`
- `llama3_2_3b`: `outputs/sink_hijacking_attack/repeated_bos_prefix/llama3_2_3b_20260714/repeated_bos_prefix_metrics.csv`
- `qwen3_4b`: `outputs/sink_hijacking_attack/repeated_bos_prefix/qwen3_4b_20260714/repeated_bos_prefix_metrics.csv`
- `mistral_7b_v0_1`: `outputs/sink_hijacking_attack/repeated_bos_prefix/mistral_7b_v0_1_20260714/repeated_bos_prefix_metrics.csv`

## Summary

| model | control | metric | n pairs | mean selected-control | 95% bootstrap CI | fraction positive | sign p | normal approx p |
|---|---|---|---:|---:|---|---:|---:|---:|
| pythia_1b | bos_repeat_only | ppl_ratio_vs_clean | 7 | -0.00221371 | [-0.0044231, -4.09113e-05] | 0.143 | 0.125 | 0.0748565 |
| pythia_1b | bos_repeat_only | inserted_attention_mean | 7 | 0.062229 | [0.0287912, 0.100666] | 1 | 0.015625 | 0.00199769 |
| pythia_1b | bos_repeat_only | inserted_minus_bos_mean | 7 | 0.119963 | [0.0510157, 0.197492] | 1 | 0.015625 | 0.00363232 |
| pythia_1b | ordinary_repeat_only | ppl_ratio_vs_clean | 7 | 0.18507 | [0.0588872, 0.344029] | 1 | 0.015625 | 0.0194089 |
| pythia_1b | ordinary_repeat_only | inserted_attention_mean | 7 | 0.353131 | [0.321316, 0.382359] | 1 | 0.015625 | 0 |
| pythia_1b | ordinary_repeat_only | inserted_minus_bos_mean | 7 | 0.636468 | [0.559681, 0.699956] | 1 | 0.015625 | 0 |
| pythia_1b | random_copy_bos | ppl_ratio_vs_clean | 7 | -0.00255109 | [-0.00446786, -0.000717844] | 0.143 | 0.125 | 0.0131128 |
| pythia_1b | random_copy_bos | inserted_attention_mean | 7 | 0.0623023 | [0.0281111, 0.100981] | 1 | 0.015625 | 0.00215964 |
| pythia_1b | random_copy_bos | inserted_minus_bos_mean | 7 | 0.12038 | [0.0506295, 0.200058] | 1 | 0.015625 | 0.00378772 |
| llama3_2_3b | bos_repeat_only | ppl_ratio_vs_clean | 4 | 0.915375 | [0.293272, 1.78768] | 1 | 0.125 | 0.0492385 |
| llama3_2_3b | bos_repeat_only | inserted_attention_mean | 4 | 0.0346958 | [0.0192591, 0.0537919] | 1 | 0.125 | 0.00116341 |
| llama3_2_3b | bos_repeat_only | inserted_minus_bos_mean | 4 | 0.0675669 | [0.0337881, 0.108648] | 1 | 0.125 | 0.00355 |
| llama3_2_3b | ordinary_repeat_only | ppl_ratio_vs_clean | 4 | 6.48197 | [1.01578, 15.5852] | 1 | 0.125 | 0.163992 |
| llama3_2_3b | ordinary_repeat_only | inserted_attention_mean | 4 | 0.689452 | [0.658797, 0.72814] | 1 | 0.125 | 0 |
| llama3_2_3b | ordinary_repeat_only | inserted_minus_bos_mean | 4 | 1.31136 | [1.27921, 1.35129] | 1 | 0.125 | 0 |
| llama3_2_3b | random_copy_bos | ppl_ratio_vs_clean | 4 | 0.907978 | [0.292833, 1.76478] | 1 | 0.125 | 0.0475605 |
| llama3_2_3b | random_copy_bos | inserted_attention_mean | 4 | 0.034684 | [0.0192479, 0.0537833] | 1 | 0.125 | 0.00116806 |
| llama3_2_3b | random_copy_bos | inserted_minus_bos_mean | 4 | 0.0675563 | [0.0337755, 0.108645] | 1 | 0.125 | 0.00355835 |
| qwen3_4b | bos_repeat_only | ppl_ratio_vs_clean | 4 | 0.0043161 | [0.00295048, 0.00590146] | 1 | 0.125 | 2.13181e-07 |
| qwen3_4b | bos_repeat_only | inserted_attention_mean | 4 | 0.0297559 | [0.0161846, 0.0457241] | 1 | 0.125 | 0.00115929 |
| qwen3_4b | bos_repeat_only | inserted_minus_bos_mean | 4 | 0.055773 | [0.0273901, 0.0893109] | 1 | 0.125 | 0.00366483 |
| qwen3_4b | ordinary_repeat_only | ppl_ratio_vs_clean | 4 | 0.0409869 | [0.00487786, 0.0822452] | 0.75 | 0.625 | 0.0883135 |
| qwen3_4b | ordinary_repeat_only | inserted_attention_mean | 4 | 0.510209 | [0.495762, 0.524655] | 1 | 0.125 | 0 |
| qwen3_4b | ordinary_repeat_only | inserted_minus_bos_mean | 4 | 0.939306 | [0.925879, 0.952733] | 1 | 0.125 | 0 |
| qwen3_4b | random_copy_bos | ppl_ratio_vs_clean | 4 | 0.00409326 | [0.0028117, 0.00564769] | 1 | 0.125 | 7.4125e-07 |
| qwen3_4b | random_copy_bos | inserted_attention_mean | 4 | 0.0297369 | [0.0161503, 0.0456933] | 1 | 0.125 | 0.00117126 |
| qwen3_4b | random_copy_bos | inserted_minus_bos_mean | 4 | 0.0557571 | [0.0273554, 0.0892966] | 1 | 0.125 | 0.00368732 |
| mistral_7b_v0_1 | bos_repeat_only | ppl_ratio_vs_clean | 4 | 1.36069 | [0.433163, 2.29099] | 1 | 0.125 | 0.0206082 |
| mistral_7b_v0_1 | bos_repeat_only | inserted_attention_mean | 4 | 0.0225857 | [0.0126867, 0.0346161] | 1 | 0.125 | 0.000877051 |
| mistral_7b_v0_1 | bos_repeat_only | inserted_minus_bos_mean | 4 | 0.0322032 | [0.0150404, 0.0529625] | 1 | 0.125 | 0.00611136 |
| mistral_7b_v0_1 | ordinary_repeat_only | ppl_ratio_vs_clean | 4 | 5.79762 | [0.882934, 13.1461] | 1 | 0.125 | 0.133216 |
| mistral_7b_v0_1 | ordinary_repeat_only | inserted_attention_mean | 4 | 0.634088 | [0.601343, 0.67182] | 1 | 0.125 | 0 |
| mistral_7b_v0_1 | ordinary_repeat_only | inserted_minus_bos_mean | 4 | 1.06714 | [1.02261, 1.11206] | 1 | 0.125 | 0 |
| mistral_7b_v0_1 | random_copy_bos | ppl_ratio_vs_clean | 4 | 1.36092 | [0.43319, 2.29184] | 1 | 0.125 | 0.020634 |
| mistral_7b_v0_1 | random_copy_bos | inserted_attention_mean | 4 | 0.0225877 | [0.0126898, 0.0346194] | 1 | 0.125 | 0.000876008 |
| mistral_7b_v0_1 | random_copy_bos | inserted_minus_bos_mean | 4 | 0.0322043 | [0.0150436, 0.0529638] | 1 | 0.125 | 0.00610627 |
| pooled_model_count | bos_repeat_only | ppl_ratio_vs_clean | 19 | 0.479264 | [0.151609, 0.899369] | 0.684 | 0.167068 | 0.0144787 |
| pooled_model_count | bos_repeat_only | inserted_attention_mean | 19 | 0.0412501 | [0.0268215, 0.0588517] | 1 | 3.8147e-06 | 1.49395e-06 |
| pooled_model_count | bos_repeat_only | inserted_minus_bos_mean | 19 | 0.0769429 | [0.0470658, 0.114004] | 1 | 3.8147e-06 | 1.37161e-05 |
| pooled_model_count | ordinary_repeat_only | ppl_ratio_vs_clean | 19 | 2.66199 | [0.541128, 5.53403] | 0.947 | 7.62939e-05 | 0.0458058 |
| pooled_model_count | ordinary_repeat_only | inserted_attention_mean | 19 | 0.516153 | [0.452443, 0.579223] | 1 | 3.8147e-06 | 0 |
| pooled_model_count | ordinary_repeat_only | inserted_minus_bos_mean | 19 | 0.932975 | [0.812666, 1.05071] | 1 | 3.8147e-06 | 0 |
| pooled_model_count | random_copy_bos | ppl_ratio_vs_clean | 19 | 0.477584 | [0.141652, 0.883771] | 0.684 | 0.167068 | 0.0144573 |
| pooled_model_count | random_copy_bos | inserted_attention_mean | 19 | 0.0412711 | [0.026324, 0.0596911] | 1 | 3.8147e-06 | 1.72034e-06 |
| pooled_model_count | random_copy_bos | inserted_minus_bos_mean | 19 | 0.077091 | [0.0468225, 0.11478] | 1 | 3.8147e-06 | 1.50243e-05 |

## Interpretation Notes

- This is sufficient for a count-sweep amplification check on the preserved artifacts.
- It is not a per-window uncertainty estimate; that would require rerunning the attack runner with per-window outputs or modifying the runner to emit them.
