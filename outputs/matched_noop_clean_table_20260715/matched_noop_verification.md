# Matched No-Op Table Verification

Verified artifacts:

- Primary table: `outputs/matched_noop_clean_table_20260715/matched_noop_clean_table.md`
- Primary model summary: `outputs/matched_noop_clean_table_20260715/matched_noop_model_summary.csv`
- Primary sequence effects: `outputs/matched_noop_clean_table_20260715/matched_noop_sequence_effects.csv`
- Early-ordinary sensitivity table: `outputs/matched_noop_clean_table_20260715_early_control/matched_noop_clean_table.md`

Matching invariants:

- Pairing key is sequence index, query position, layer, head, head kind, and delta.
- Within each pair, the only changed field is target kind: sink token versus ordinary token.
- The fixed intervention mass is `delta=0.05` for every primary pair.
- Effects are averaged at the sequence level before confidence intervals are computed.
- Skipped pairs are excluded only when either paired row could not receive the fixed mass.

Model inputs:

- `llama3_2_3b`: `outputs/causal_noop_full_wikitext_20260714/llama3_2_3b/wikitext_test/site_results.csv`
- `mistral_7b_v0_1`: `outputs/causal_noop_mistral_wikitext_fixed_20260714/mistral_7b_v0_1/wikitext_test/site_results.csv`
- `pythia_1b`: `outputs/causal_noop_full_wikitext_20260714/pythia_1b/wikitext_test/site_results.csv`

Important scope note:

The representative table uses Pythia-1B, Llama-3.2-3B, and the separate fixed Mistral run. The combined full Wikitext directory contains all-zero Mistral site effects, so it is not used for the Mistral row.

Completion judgment:

The clean matched table is sufficient to evaluate the No-Op mechanism claim. It does not provide strong positive ΔNLL evidence because all representative ΔNLL confidence intervals cross zero.
