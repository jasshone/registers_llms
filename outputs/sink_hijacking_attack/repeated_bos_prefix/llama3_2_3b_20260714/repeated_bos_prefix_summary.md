# Repeated-BOS Prefix Attack

Clean PPL: 8.963; clean BOS attention mean: 0.6274.

Mechanism checks:

- Pearson r(inserted attention, PPL ratio) across non-clean rows: 0.4824.
- Pearson r(inserted-minus-BOS attention, PPL ratio) across non-clean rows: 0.4429.
- Strongest selected-copy count 64: PPL ratio 21.3277 vs layer-matched random 19.1000; inserted attention 0.7528 vs 0.7371.
- Same count ordinary-token repeat: PPL ratio 0.9963; inserted attention 0.0036.

| count | condition | inserted attn | BOS attn | inserted-BOS | PPL ratio |
|---:|---|---:|---:|---:|---:|
| 8 | bos_repeat_only | 0.5909 | 0.0739 | +0.5171 | 1.4681 |
| 8 | ordinary_repeat_only | 0.0024 | 0.6260 | -0.6236 | 0.9927 |
| 8 | selected_copy_bos | 0.6551 | 0.0070 | +0.6481 | 1.6776 |
| 8 | random_copy_bos | 0.5910 | 0.0739 | +0.5171 | 1.4680 |
| 16 | bos_repeat_only | 0.6315 | 0.0395 | +0.5920 | 1.9629 |
| 16 | ordinary_repeat_only | 0.0027 | 0.6258 | -0.6231 | 0.9933 |
| 16 | selected_copy_bos | 0.6677 | 0.0039 | +0.6637 | 2.3400 |
| 16 | random_copy_bos | 0.6315 | 0.0395 | +0.5920 | 1.9638 |
| 32 | bos_repeat_only | 0.6713 | 0.0210 | +0.6503 | 3.7416 |
| 32 | ordinary_repeat_only | 0.0031 | 0.6254 | -0.6223 | 0.9937 |
| 32 | selected_copy_bos | 0.6941 | 0.0022 | +0.6919 | 4.5587 |
| 32 | random_copy_bos | 0.6712 | 0.0210 | +0.6503 | 3.7401 |
| 64 | bos_repeat_only | 0.7371 | 0.0115 | +0.7256 | 19.0698 |
| 64 | ordinary_repeat_only | 0.0036 | 0.6248 | -0.6213 | 0.9963 |
| 64 | selected_copy_bos | 0.7528 | 0.0012 | +0.7515 | 21.3277 |
| 64 | random_copy_bos | 0.7371 | 0.0115 | +0.7256 | 19.1000 |
