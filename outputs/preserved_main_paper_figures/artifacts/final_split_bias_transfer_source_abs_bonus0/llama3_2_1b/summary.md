# llama3_2_1b Final Split Bias Transfer

## Test Selection

| mask | k | scale | layers | relocation | PPL ratio | PPL change |
|---|---:|---:|---|---:|---:|---:|
| bos_only | 4 | 1 | 0:2 1:2 | 0.396598 | 1.09015 | +9.01% |

## Top Validation Transfer Rows

| mask | k | scale | layers | relocation | PPL ratio |
|---|---:|---:|---|---:|---:|
| bos_only | 16 | 3 | 0:4 1:10 2:1 3:1 | 0.0104145 | 1.06141 |
| bos_only | 2 | 2 | 0:1 1:1 | -0.519065 | 1.06386 |
| pipeline_sinks_plus_bos | 16 | 3 | 0:4 1:10 2:1 3:1 | 0.0104231 | 1.06426 |
| pipeline_sinks_plus_bos | 2 | 2 | 0:1 1:1 | -0.518982 | 1.06626 |
| bos_only | 8 | 3 | 0:2 1:6 | 0.0135154 | 1.07022 |
| pipeline_sinks_plus_bos | 8 | 3 | 0:2 1:6 | 0.0135141 | 1.0728 |
| bos_only | 4 | 3 | 0:2 1:2 | 0.0160757 | 1.07299 |
| pipeline_sinks_plus_bos | 4 | 3 | 0:2 1:2 | 0.0160673 | 1.07554 |
| bos_only | 4 | 1 | 0:2 1:2 | 0.413986 | 1.09357 |
| bos_only | 8 | 1 | 0:2 1:6 | 0.411683 | 1.09439 |
