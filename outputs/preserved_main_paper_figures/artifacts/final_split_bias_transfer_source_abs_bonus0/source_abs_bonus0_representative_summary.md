# Source Abs Bonus 0 Representative Results

Candidate score used for these reruns:

`score = mean_source - mean_control + 0 * abs(mean_source)`

Most rows use the final-split runner with 16/16/16 train/val/test windows. Gemma uses the Gemma two-scale runner on train16 because Gemma is outside the final-split config.

| family | model | mask | k | scale | layers | test/heldout reloc | PPL ratio | status |
|---|---|---|---:|---|---|---:|---:|---|
| gpt2 | gpt2 | bos_only | 4 | 3.0 | `1:1 2:3` | 0.3006 | 1.0024 | ok |
| pythia | pythia_410m | bos_only | 4 | 3.0 | `5:4` | 0.3159 | 1.0178 | ok |
| llama | llama3_2_1b | bos_only | 4 | 1.0 | `0:2 1:2` | 0.3966 | 1.0901 | ok |
| qwen3 | qwen3_1_7b | bos_only | 1 | 3.0 | `2:1` | 0.4477 | 0.9768 | ok |
| opt | opt_125m | bos_only | 8 | 3.0 | `0:8` | 0.4428 | 1.0092 | ok |
| mistral | mistral_7b_v0_1 | bos_only | 1 | 2.0 | `1:1` | 0.3383 | 0.9943 | ok |
| phi | phi_1_5 | bos_only | 2 | 3.0 | `0:1 1:1` | 0.4921 | 1.0023 | ok |
| gemma | unsloth_gemma3_4b_pt | pipeline_sinks_plus_bos_layers_0_1_2 | 32 | src=8, dummy=16 | `0:15 2:1` | 0.3022 | 0.9374 | ok |

Note: `qwen3_0_6b` was also tried and failed validation selection: ValueError('no positive validation relocation; best relocation=-0.000163391, ppl_ratio=23.8065, mask=bos_only, topk=8, scale=1.25').
