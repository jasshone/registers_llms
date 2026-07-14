# Intervention Decomposition Summary: llama3_2_3b

| condition | group | reloc | PPL ratio | KL | BOS attn | dummy attn |
|---|---|---:|---:|---:|---:|---:|
| dummy_add_only | selected | -0.0733503 | 1.01837 | 0.032823 | 0.363159 | 0.289809 |
| dummy_only | selected | -0.626408 | 0.99022 | 0.0151024 | 0.628357 | 0.00194822 |
| full_transfer | selected | 0.529108 | 1.03526 | 0.0593778 | 0.0402216 | 0.56933 |
| random_dummy_add_only | layer_matched_random | -0.626424 | 0.99013 | 0.0152606 | 0.62839 | 0.00196604 |
| random_full_transfer | layer_matched_random | -0.626516 | 0.989846 | 0.0151751 | 0.628473 | 0.00195779 |
| source_subtract_only | selected | -0.0462639 | 20.1929 | 3.04792 | 0.0498484 | 0.00358447 |
