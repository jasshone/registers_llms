# mistral_7b_v0_1 Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 1 | 2 | 1:1 | 0.338345 | 0.994259 | -0.57% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| bos_only | 16 | 1.5 | 1:16 | 0.265472 | 0.99006 |
| bos_only | 1 | 2 | 1:1 | 0.350646 | 0.990203 |
| bos_only | 16 | 2 | 1:16 | 0.267925 | 0.990276 |
| bos_only | 1 | 3 | 1:1 | 0.354029 | 0.990742 |
| bos_only | 16 | 3 | 1:16 | 0.269042 | 0.991481 |
| bos_only | 1 | 1.5 | 1:1 | 0.341319 | 0.992213 |
| bos_only | 16 | 1.25 | 1:16 | 0.265432 | 0.99246 |
| pipeline_sinks_plus_bos | 1 | 3 | 1:1 | 0.511197 | 0.992924 |
| bos_only | 4 | 2 | 1:4 | 0.291697 | 0.993027 |
| bos_only | 2 | 2 | 1:2 | 0.281387 | 0.993389 |
