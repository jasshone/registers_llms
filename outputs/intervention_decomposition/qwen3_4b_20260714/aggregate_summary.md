# Intervention Decomposition Summary: qwen3_4b

| condition | group | reloc | PPL ratio | KL | BOS attn | dummy attn |
|---|---|---:|---:|---:|---:|---:|
| dummy_add_only | selected | 0.00022472 | 0.959981 | 0.0162489 | 0.231374 | 0.231599 |
| dummy_only | selected | 0.000178395 | 0.960512 | 0.0163329 | 0.231383 | 0.231561 |
| full_transfer | selected | 0.419753 | 0.965831 | 0.0150313 | 0.00326215 | 0.423015 |
| random_dummy_add_only | layer_matched_random | 0.000179938 | 0.960052 | 0.0163329 | 0.231397 | 0.231577 |
| random_full_transfer | layer_matched_random | 0.000206466 | 0.960207 | 0.0163998 | 0.231413 | 0.231619 |
| source_subtract_only | selected | 0.419123 | 0.967925 | 0.0147724 | 0.00326165 | 0.422385 |
