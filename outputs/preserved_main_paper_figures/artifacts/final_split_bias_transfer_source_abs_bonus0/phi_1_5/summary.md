# phi_1_5 Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 2 | 3 | 0:1 1:1 | 0.492134 | 1.00228 | +0.23% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| pipeline_sinks_plus_bos | 1 | 1.5 | 1:1 | -0.517892 | 0.987996 |
| bos_only | 1 | 1.25 | 1:1 | -0.516366 | 0.988149 |
| pipeline_sinks_plus_bos | 1 | 1.25 | 1:1 | -0.518009 | 0.988315 |
| pipeline_sinks_plus_bos | 2 | 1 | 0:1 1:1 | -0.521521 | 0.988323 |
| bos_only | 1 | 1.5 | 1:1 | -0.515529 | 0.988337 |
| bos_only | 1 | 1 | 1:1 | -0.517066 | 0.988571 |
| bos_only | 1 | 2 | 1:1 | -0.511909 | 0.988878 |
| bos_only | 1 | 3 | 1:1 | -0.502343 | 0.989062 |
| pipeline_sinks_plus_bos | 1 | 1 | 1:1 | -0.517879 | 0.989224 |
| pipeline_sinks_plus_bos | 1 | 2 | 1:1 | -0.517022 | 0.989606 |
