# Repeated-BOS Prefix Attack

Clean PPL: 15.85; clean BOS attention mean: 0.3453.

Mechanism checks:

- Pearson r(inserted attention, PPL ratio) across non-clean rows: 0.6945.
- Pearson r(inserted-minus-BOS attention, PPL ratio) across non-clean rows: 0.6298.
- Strongest selected-copy count 64: PPL ratio 1.7541 vs layer-matched random 1.7613; inserted attention 0.5162 vs 0.5062.
- Same count ordinary-token repeat: PPL ratio 1.1710; inserted attention 0.2412.

| count | condition | inserted attn | BOS attn | inserted-BOS | PPL ratio |
|---:|---|---:|---:|---:|---:|
| 1 | bos_repeat_only | 0.1831 | 0.1830 | +0.0001 | 1.0066 |
| 1 | ordinary_repeat_only | 0.0020 | 0.3447 | -0.3427 | 0.9952 |
| 1 | selected_copy_bos | 0.3370 | 0.0268 | +0.3101 | 1.0093 |
| 1 | random_copy_bos | 0.1827 | 0.1838 | -0.0011 | 1.0078 |
| 2 | bos_repeat_only | 0.2517 | 0.1258 | +0.1259 | 1.0171 |
| 2 | ordinary_repeat_only | 0.0032 | 0.3442 | -0.3410 | 0.9922 |
| 2 | selected_copy_bos | 0.3624 | 0.0191 | +0.3434 | 1.0168 |
| 2 | random_copy_bos | 0.2505 | 0.1271 | +0.1234 | 1.0182 |
| 4 | bos_repeat_only | 0.3138 | 0.0784 | +0.2355 | 1.0406 |
| 4 | ordinary_repeat_only | 0.0049 | 0.3428 | -0.3379 | 0.9919 |
| 4 | selected_copy_bos | 0.3851 | 0.0128 | +0.3723 | 1.0396 |
| 4 | random_copy_bos | 0.3139 | 0.0784 | +0.2355 | 1.0413 |
| 8 | bos_repeat_only | 0.3660 | 0.0456 | +0.3203 | 1.0926 |
| 8 | ordinary_repeat_only | 0.0094 | 0.3388 | -0.3294 | 0.9946 |
| 8 | selected_copy_bos | 0.4097 | 0.0082 | +0.4015 | 1.0907 |
| 8 | random_copy_bos | 0.3657 | 0.0458 | +0.3198 | 1.0936 |
| 16 | bos_repeat_only | 0.4116 | 0.0256 | +0.3860 | 1.1972 |
| 16 | ordinary_repeat_only | 0.0447 | 0.2998 | -0.2552 | 1.0015 |
| 16 | selected_copy_bos | 0.4395 | 0.0049 | +0.4346 | 1.1950 |
| 16 | random_copy_bos | 0.4118 | 0.0255 | +0.3863 | 1.1966 |
| 32 | bos_repeat_only | 0.4566 | 0.0141 | +0.4425 | 1.3907 |
| 32 | ordinary_repeat_only | 0.1467 | 0.2237 | -0.0770 | 1.0482 |
| 32 | selected_copy_bos | 0.4740 | 0.0028 | +0.4712 | 1.3848 |
| 32 | random_copy_bos | 0.4572 | 0.0142 | +0.4430 | 1.3891 |
| 64 | bos_repeat_only | 0.5056 | 0.0077 | +0.4979 | 1.7609 |
| 64 | ordinary_repeat_only | 0.2412 | 0.1654 | +0.0759 | 1.1710 |
| 64 | selected_copy_bos | 0.5162 | 0.0015 | +0.5148 | 1.7541 |
| 64 | random_copy_bos | 0.5062 | 0.0078 | +0.4984 | 1.7613 |
