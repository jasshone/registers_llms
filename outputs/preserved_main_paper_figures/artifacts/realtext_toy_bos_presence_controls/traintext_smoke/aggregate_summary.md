# Toy BOS Presence Controls on Wikitext Train Text

Toy LMs trained from scratch on actual Wikitext-103 train text tokenized with the Pythia-70M tokenizer. Each row aggregates final-step metrics across seeds.

Token source: `Salesforce/wikitext` / `wikitext-103-raw-v1` / split `train`; max chars `10000000`; tokens used `50000`.

| variant | BOS positions | seeds | loss | PPL | pos0 attn | control attn | pos0-control | BOS attn sum | max BOS-pos attn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_0 | 0 | 1 | 10.8245 +/- 0.0000 | 5.02e+04 +/- 0 | 0.0987 +/- 0.0000 | 0.0853 +/- 0.0000 | +0.0134 +/- 0.0000 | 0.0987 +/- 0.0000 | 0.0987 +/- 0.0000 |
| no_bos | none | 1 | 10.8277 +/- 0.0000 | 5.04e+04 +/- 0 | 0.0989 +/- 0.0000 | 0.0852 +/- 0.0000 | +0.0136 +/- 0.0000 | 0.0000 +/- 0.0000 | nan +/- 0.0000 |

Interpretation guide: `no_bos` tests whether ordinary text at absolute position 0 becomes sink-like. `single_8` and `single_32` test whether the sink follows BOS identity. Duplicate variants test whether later BOS copies compete with the first-position BOS.
