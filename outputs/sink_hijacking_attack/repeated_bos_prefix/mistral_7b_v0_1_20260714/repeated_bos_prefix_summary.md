# Repeated-BOS Prefix Attack

Clean PPL: 6.441; clean BOS attention mean: 0.4619.

Mechanism checks:

- Pearson r(inserted attention, PPL ratio) across non-clean rows: 0.5054.
- Pearson r(inserted-minus-BOS attention, PPL ratio) across non-clean rows: 0.4758.
- Strongest selected-copy count 64: PPL ratio 18.1131 vs layer-matched random 15.2443; inserted attention 0.7118 vs 0.7014.
- Same count ordinary-token repeat: PPL ratio 0.9962; inserted attention 0.0217.

| count | condition | inserted attn | BOS attn | inserted-BOS | PPL ratio |
|---:|---|---:|---:|---:|---:|
| 8 | bos_repeat_only | 0.5530 | 0.0692 | +0.4837 | 1.2204 |
| 8 | ordinary_repeat_only | 0.0060 | 0.4604 | -0.4544 | 0.9937 |
| 8 | selected_copy_bos | 0.5941 | 0.0460 | +0.5481 | 1.5256 |
| 8 | random_copy_bos | 0.5530 | 0.0692 | +0.4837 | 1.2204 |
| 16 | bos_repeat_only | 0.6002 | 0.0376 | +0.5626 | 1.6670 |
| 16 | ordinary_repeat_only | 0.0070 | 0.4605 | -0.4535 | 0.9941 |
| 16 | selected_copy_bos | 0.6240 | 0.0271 | +0.5969 | 2.2281 |
| 16 | random_copy_bos | 0.6002 | 0.0376 | +0.5626 | 1.6670 |
| 32 | bos_repeat_only | 0.6380 | 0.0200 | +0.6180 | 3.5932 |
| 32 | ordinary_repeat_only | 0.0119 | 0.4581 | -0.4462 | 0.9942 |
| 32 | selected_copy_bos | 0.6530 | 0.0163 | +0.6368 | 5.3020 |
| 32 | random_copy_bos | 0.6380 | 0.0200 | +0.6180 | 3.5934 |
| 64 | bos_repeat_only | 0.7014 | 0.0111 | +0.6904 | 15.2454 |
| 64 | ordinary_repeat_only | 0.0217 | 0.4526 | -0.4309 | 0.9962 |
| 64 | selected_copy_bos | 0.7118 | 0.0101 | +0.7016 | 18.1131 |
| 64 | random_copy_bos | 0.7014 | 0.0111 | +0.6904 | 15.2443 |
