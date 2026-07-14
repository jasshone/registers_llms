# opt_125m Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 8 | 3 | 0:8 | 0.442767 | 1.00922 | +0.92% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| bos_only | 8 | 3 | 0:8 | 0.46418 | 1.00375 |
| pipeline_sinks_plus_bos | 8 | 3 | 0:8 | 0.46418 | 1.00375 |
| bos_only | 16 | 3 | 0:9 2:7 | 0.462738 | 1.00449 |
| pipeline_sinks_plus_bos | 16 | 3 | 0:9 2:7 | 0.462738 | 1.00449 |
| bos_only | 16 | 2 | 0:9 2:7 | 0.461809 | 1.00536 |
| pipeline_sinks_plus_bos | 16 | 2 | 0:9 2:7 | 0.461809 | 1.00536 |
| bos_only | 8 | 2 | 0:8 | 0.463023 | 1.00547 |
| pipeline_sinks_plus_bos | 8 | 2 | 0:8 | 0.463023 | 1.00547 |
| bos_only | 16 | 1.5 | 0:9 2:7 | 0.460272 | 1.00854 |
| pipeline_sinks_plus_bos | 16 | 1.5 | 0:9 2:7 | 0.460272 | 1.00854 |
