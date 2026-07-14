# qwen3_1_7b Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 1 | 3 | 2:1 | 0.447713 | 0.976811 | -2.32% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| bos_only | 1 | 3 | 2:1 | 0.454981 | 0.967892 |
| pipeline_sinks_plus_bos | 2 | 3 | 2:2 | 0.455334 | 0.968017 |
| bos_only | 2 | 3 | 2:2 | 0.455322 | 0.968658 |
| pipeline_sinks_plus_bos | 4 | 3 | 2:4 | 0.455536 | 0.968929 |
| bos_only | 16 | 3 | 1:7 2:9 | 0.0106012 | 0.969335 |
| bos_only | 4 | 3 | 2:4 | 0.455448 | 0.969425 |
| pipeline_sinks_plus_bos | 8 | 3 | 1:1 2:7 | 0.45624 | 0.969515 |
| bos_only | 8 | 3 | 1:1 2:7 | 0.456306 | 0.970218 |
| pipeline_sinks_plus_bos | 1 | 3 | 2:1 | 0.455153 | 0.972023 |
| bos_only | 8 | 2 | 1:1 2:7 | 0.456208 | 0.972664 |
