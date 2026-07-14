# Repeated-BOS Prefix Attack

Clean PPL: 16.72; clean BOS attention mean: 0.4339.

Mechanism checks:

- Pearson r(inserted attention, PPL ratio) across non-clean rows: 0.5412.
- Pearson r(inserted-minus-BOS attention, PPL ratio) across non-clean rows: 0.5188.
- Strongest selected-copy count 64: PPL ratio 1.0715 vs layer-matched random 1.0652; inserted attention 0.5404 vs 0.5276.
- Same count ordinary-token repeat: PPL ratio 0.9668; inserted attention 0.0087.

| count | condition | inserted attn | BOS attn | inserted-BOS | PPL ratio |
|---:|---|---:|---:|---:|---:|
| 8 | bos_repeat_only | 0.4379 | 0.0547 | +0.3832 | 0.9652 |
| 8 | ordinary_repeat_only | 0.0035 | 0.4319 | -0.4283 | 0.9744 |
| 8 | selected_copy_bos | 0.4923 | 0.0015 | +0.4908 | 0.9692 |
| 8 | random_copy_bos | 0.4379 | 0.0547 | +0.3832 | 0.9651 |
| 16 | bos_repeat_only | 0.4750 | 0.0297 | +0.4453 | 0.9829 |
| 16 | ordinary_repeat_only | 0.0045 | 0.4311 | -0.4266 | 0.9705 |
| 16 | selected_copy_bos | 0.5072 | 0.0012 | +0.5061 | 0.9855 |
| 16 | random_copy_bos | 0.4750 | 0.0297 | +0.4453 | 0.9831 |
| 32 | bos_repeat_only | 0.5040 | 0.0157 | +0.4883 | 1.0137 |
| 32 | ordinary_repeat_only | 0.0059 | 0.4295 | -0.4237 | 0.9684 |
| 32 | selected_copy_bos | 0.5236 | 0.0009 | +0.5227 | 1.0180 |
| 32 | random_copy_bos | 0.5040 | 0.0157 | +0.4883 | 1.0144 |
| 64 | bos_repeat_only | 0.5276 | 0.0082 | +0.5194 | 1.0650 |
| 64 | ordinary_repeat_only | 0.0087 | 0.4281 | -0.4194 | 0.9668 |
| 64 | selected_copy_bos | 0.5404 | 0.0006 | +0.5397 | 1.0715 |
| 64 | random_copy_bos | 0.5276 | 0.0082 | +0.5194 | 1.0652 |
