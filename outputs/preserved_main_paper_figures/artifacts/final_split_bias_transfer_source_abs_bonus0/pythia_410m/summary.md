# pythia_410m Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 4 | 3 | 5:4 | 0.315923 | 1.01784 | +1.78% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| pipeline_sinks_plus_bos | 2 | 2 | 5:2 | -0.0705217 | 0.986494 |
| pipeline_sinks_plus_bos | 8 | 1 | 5:5 6:3 | -0.0127461 | 0.988468 |
| pipeline_sinks_plus_bos | 2 | 1 | 5:2 | -0.127745 | 0.989102 |
| pipeline_sinks_plus_bos | 8 | 1.5 | 5:5 6:3 | 0.0901721 | 0.991504 |
| pipeline_sinks_plus_bos | 16 | 1 | 4:1 5:5 6:10 | 0.0466999 | 0.991552 |
| pipeline_sinks_plus_bos | 4 | 1.5 | 5:4 | 0.00326088 | 0.992424 |
| bos_only | 4 | 1 | 5:4 | -0.00822745 | 0.992656 |
| pipeline_sinks_plus_bos | 4 | 1 | 5:4 | -0.0529276 | 0.993276 |
| pipeline_sinks_plus_bos | 1 | 3 | 5:1 | -0.0384236 | 0.993454 |
| pipeline_sinks_plus_bos | 4 | 1.25 | 5:4 | -0.0280514 | 0.99347 |
