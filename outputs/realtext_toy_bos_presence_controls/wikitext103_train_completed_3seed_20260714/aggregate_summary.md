# Completed Toy BOS Presence Controls

Merged final-step metrics for all six Figure 3 identity-control variants with three seeds each. Raw per-step metrics remain in the source run directories listed in `source_manifest.json`.

| variant | BOS positions | seeds | pos0 attn | total BOS attn | loss | PPL |
|---|---|---:|---:|---:|---:|---:|
| single_0 | 0 | 0 1 2 | 0.082983 +/- 0.000946 | 0.082983 +/- 0.000946 | 6.296891 +/- 0.004336 | 542.885 +/- 2.35533 |
| no_bos | none | 0 1 2 | 0.025055 +/- 0.000475 | 0.000000 +/- 0.000000 | 6.262408 +/- 0.013699 | 524.513 +/- 7.19998 |
| single_8 | 8 | 0 1 2 | 0.022242 +/- 0.000354 | 0.013789 +/- 0.002536 | 6.232605 +/- 0.017729 | 509.133 +/- 8.99172 |
| single_32 | 32 | 0 1 2 | 0.022276 +/- 0.000462 | 0.005733 +/- 0.000692 | 6.209984 +/- 0.038407 | 497.937 +/- 19.0722 |
| duplicate_0,8 | 0 8 | 0 1 2 | 0.081547 +/- 0.006644 | 0.094246 +/- 0.007667 | 6.245222 +/- 0.034094 | 515.744 +/- 17.7077 |
| duplicate_0,32 | 0 32 | 0 1 2 | 0.067696 +/- 0.007633 | 0.074619 +/- 0.007833 | 6.243074 +/- 0.015514 | 514.479 +/- 8.01664 |
