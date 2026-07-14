# Intervention Decomposition Summary: pythia_1b

| condition | group | reloc | PPL ratio | KL | BOS attn | dummy attn |
|---|---|---:|---:|---:|---:|---:|
| dummy_add_only | selected | 0.0103226 | 1.00784 | 0.0134882 | 0.182677 | 0.193 |
| dummy_only | selected | -0.348284 | 0.996428 | 0.0119891 | 0.351206 | 0.00292284 |
| full_transfer | selected | 0.313848 | 1.0072 | 0.0120121 | 0.0226395 | 0.336488 |
| random_dummy_add_only | layer_matched_random | -0.227572 | 0.994829 | 0.0126379 | 0.294767 | 0.0671947 |
| random_full_transfer | layer_matched_random | -0.229975 | 0.997057 | 0.0141138 | 0.296036 | 0.066061 |
| source_subtract_only | selected | -0.159189 | 1.43428 | 0.367771 | 0.161787 | 0.00259775 |
