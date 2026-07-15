# Matched No-Op Intervention Audit

Command verified existing matched attention-mass No-Op outputs for Pythia-1B, Llama-3.2-3B, and Qwen3-4B on Wikitext test:

```bash
.conda-env/bin/python audit_causal_noop_outputs.py --out outputs/causal_noop_full_wikitext_20260714 --models pythia_1b llama3_2_3b qwen3_4b --datasets wikitext_test --deltas 0.02 0.05 0.10 --expected-heads 16 --min-sequences 128 --primary-delta 0.05 --sensitivity-min-sequences 50 --value-min-sequences 100 --secondary-min-sequences 50
```

Result: audit passed. The only warning was `llama3_2_3b/wikitext_test: no primary rows for middle layer band`; all-layer primary matched sink-vs-ordinary comparisons, delta sensitivity, value validation, and secondary all-head artifacts were present at the requested thresholds.

Interpretation: the matched No-Op item is covered if the paper keeps No-Op as a central claim, with the caveat that Wikitext-test coverage is 128+ paired sequences rather than 256+ for every model.
