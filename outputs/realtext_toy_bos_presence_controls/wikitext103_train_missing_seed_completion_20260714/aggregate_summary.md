# Toy BOS Presence Controls on Wikitext Train Text

Toy LMs trained from scratch on actual Wikitext-103 train text tokenized with the Pythia-70M tokenizer. Each row aggregates final-step metrics across seeds.

Token source: `Salesforce/wikitext` / `wikitext-103-raw-v1` / split `train`; max chars `10000000`; tokens used `2205403`.

| variant | BOS positions | seeds | loss | PPL | pos0 attn | control attn | pos0-control | BOS attn sum | max BOS-pos attn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_bos | none | 2 | 6.2638 +/- 0.0191 | 525 +/- 10 | 0.0253 +/- 0.0003 | 0.0167 +/- 0.0005 | +0.0086 +/- 0.0002 | 0.0000 +/- 0.0000 | nan +/- 0.0000 |
| single_8 | 8 | 1 | 6.2382 +/- 0.0000 | 512 +/- 0 | 0.0219 +/- 0.0000 | 0.0185 +/- 0.0000 | +0.0034 +/- 0.0000 | 0.0138 +/- 0.0000 | 0.0138 +/- 0.0000 |
| single_32 | 32 | 1 | 6.1700 +/- 0.0000 | 478 +/- 0 | 0.0222 +/- 0.0000 | 0.0178 +/- 0.0000 | +0.0044 +/- 0.0000 | 0.0054 +/- 0.0000 | 0.0054 +/- 0.0000 |
| duplicate_0,8 | 0 8 | 1 | 6.2346 +/- 0.0000 | 510 +/- 0 | 0.0751 +/- 0.0000 | 0.0134 +/- 0.0000 | +0.0618 +/- 0.0000 | 0.0865 +/- 0.0000 | 0.0751 +/- 0.0000 |
| duplicate_0,32 | 0 32 | 1 | 6.2610 +/- 0.0000 | 524 +/- 0 | 0.0623 +/- 0.0000 | 0.0129 +/- 0.0000 | +0.0493 +/- 0.0000 | 0.0700 +/- 0.0000 | 0.0623 +/- 0.0000 |

Interpretation guide: `no_bos` tests whether ordinary text at absolute position 0 becomes sink-like. `single_8` and `single_32` test whether the sink follows BOS identity. Duplicate variants test whether later BOS copies compete with the first-position BOS.
