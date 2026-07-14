# Toy BOS Presence Controls on Wikitext Train Text

Toy LMs trained from scratch on actual Wikitext-103 train text tokenized with the Pythia-70M tokenizer. Each row aggregates final-step metrics across seeds.

Token source: `Salesforce/wikitext` / `wikitext-103-raw-v1` / split `train`; max chars `10000000`; tokens used `2205403`.

| variant | BOS positions | seeds | loss | PPL | pos0 attn | control attn | pos0-control | BOS attn sum | max BOS-pos attn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_0 | 0 | 2 | 6.2946 +/- 0.0023 | 542 +/- 1.26 | 0.0829 +/- 0.0013 | 0.0129 +/- 0.0002 | +0.0699 +/- 0.0015 | 0.0829 +/- 0.0013 | 0.0829 +/- 0.0013 |
| no_bos | none | 1 | 6.2596 +/- 0.0000 | 523 +/- 0 | 0.0246 +/- 0.0000 | 0.0154 +/- 0.0000 | +0.0092 +/- 0.0000 | 0.0000 +/- 0.0000 | nan +/- 0.0000 |
| single_8 | 8 | 1 | 6.2127 +/- 0.0000 | 499 +/- 0 | 0.0223 +/- 0.0000 | 0.0176 +/- 0.0000 | +0.0047 +/- 0.0000 | 0.0113 +/- 0.0000 | 0.0113 +/- 0.0000 |
| single_32 | 32 | 1 | 6.2134 +/- 0.0000 | 499 +/- 0 | 0.0219 +/- 0.0000 | 0.0176 +/- 0.0000 | +0.0043 +/- 0.0000 | 0.0065 +/- 0.0000 | 0.0065 +/- 0.0000 |
| duplicate_0,8 | 0 8 | 1 | 6.2177 +/- 0.0000 | 502 +/- 0 | 0.0884 +/- 0.0000 | 0.0140 +/- 0.0000 | +0.0744 +/- 0.0000 | 0.1018 +/- 0.0000 | 0.0884 +/- 0.0000 |
| duplicate_0,32 | 0 32 | 1 | 6.2333 +/- 0.0000 | 509 +/- 0 | 0.0764 +/- 0.0000 | 0.0132 +/- 0.0000 | +0.0632 +/- 0.0000 | 0.0837 +/- 0.0000 | 0.0764 +/- 0.0000 |

Interpretation guide: `no_bos` tests whether ordinary text at absolute position 0 becomes sink-like. `single_8` and `single_32` test whether the sink follows BOS identity. Duplicate variants test whether later BOS copies compete with the first-position BOS.
