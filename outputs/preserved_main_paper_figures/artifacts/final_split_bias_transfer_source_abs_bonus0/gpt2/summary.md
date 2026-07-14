# gpt2 Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 4 | 3 | 1:1 2:3 | 0.30064 | 1.00237 | +0.24% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| bos_only | 4 | 3 | 1:1 2:3 | 0.322359 | 1.00164 |
| pipeline_sinks_plus_bos | 4 | 3 | 1:1 2:3 | 0.322341 | 1.00212 |
| bos_only | 4 | 2 | 1:1 2:3 | 0.305707 | 1.00285 |
| pipeline_sinks_plus_bos | 4 | 2 | 1:1 2:3 | 0.30569 | 1.0033 |
| bos_only | 1 | 1 | 2:1 | -0.277738 | 1.00815 |
| pipeline_sinks_plus_bos | 1 | 1 | 2:1 | -0.277735 | 1.00846 |
| bos_only | 1 | 1.5 | 2:1 | -0.269735 | 1.00996 |
| bos_only | 1 | 1.25 | 2:1 | -0.272741 | 1.01019 |
| pipeline_sinks_plus_bos | 1 | 1.5 | 2:1 | -0.269728 | 1.01044 |
| pipeline_sinks_plus_bos | 1 | 1.25 | 2:1 | -0.272735 | 1.01055 |
