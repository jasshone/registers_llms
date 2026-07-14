# Toy BOS Presence Controls on Wikitext Train Text

Toy LMs trained from scratch on actual Wikitext-103 train text tokenized with the Pythia-70M tokenizer. Each row aggregates final-step metrics across seeds.

Token source: `Salesforce/wikitext` / `wikitext-103-raw-v1` / split `train`; max chars `10000000`; tokens used `2205403`.

| variant | BOS positions | seeds | loss | PPL | pos0 attn | control attn | pos0-control | BOS attn sum | max BOS-pos attn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_8 | 8 | 1 | 6.2468 +/- 0.0000 | 516 +/- 0 | 0.0226 +/- 0.0000 | 0.0182 +/- 0.0000 | +0.0043 +/- 0.0000 | 0.0163 +/- 0.0000 | 0.0163 +/- 0.0000 |
| single_32 | 32 | 1 | 6.2466 +/- 0.0000 | 516 +/- 0 | 0.0228 +/- 0.0000 | 0.0192 +/- 0.0000 | +0.0036 +/- 0.0000 | 0.0053 +/- 0.0000 | 0.0053 +/- 0.0000 |
| duplicate_0,8 | 0 8 | 1 | 6.2834 +/- 0.0000 | 536 +/- 0 | 0.0811 +/- 0.0000 | 0.0139 +/- 0.0000 | +0.0672 +/- 0.0000 | 0.0944 +/- 0.0000 | 0.0811 +/- 0.0000 |
| duplicate_0,32 | 0 32 | 1 | 6.2350 +/- 0.0000 | 510 +/- 0 | 0.0644 +/- 0.0000 | 0.0136 +/- 0.0000 | +0.0508 +/- 0.0000 | 0.0702 +/- 0.0000 | 0.0644 +/- 0.0000 |

Interpretation guide: `no_bos` tests whether ordinary text at absolute position 0 becomes sink-like. `single_8` and `single_32` test whether the sink follows BOS identity. Duplicate variants test whether later BOS copies compete with the first-position BOS.
