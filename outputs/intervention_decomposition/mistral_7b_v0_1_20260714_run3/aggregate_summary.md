# Intervention Decomposition Summary: mistral_7b_v0_1

| condition | group | reloc | PPL ratio | KL | BOS attn | dummy attn |
|---|---|---:|---:|---:|---:|---:|
| dummy_add_only | selected | -0.08981 | 0.998874 | 0.00764513 | 0.308076 | 0.218266 |
| dummy_only | selected | -0.462931 | 0.991213 | 0.0121221 | 0.465051 | 0.00211981 |
| full_transfer | selected | 0.348536 | 0.992327 | 0.0181832 | 0.0348263 | 0.383363 |
| random_dummy_add_only | layer_matched_random | -0.462924 | 0.990484 | 0.0114768 | 0.465029 | 0.00210442 |
| random_full_transfer | layer_matched_random | -0.462998 | 0.990567 | 0.0114603 | 0.465099 | 0.00210096 |
| source_subtract_only | selected | -0.0470177 | 1.24295 | 0.245963 | 0.0496831 | 0.00266541 |
