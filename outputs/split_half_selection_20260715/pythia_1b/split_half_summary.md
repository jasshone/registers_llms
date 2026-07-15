# Split-Half Neuron Selection Robustness

Pythia-1B BOS-only source-mask check. Two disjoint 64-window Wikitext halves select top-16 neurons independently; each selected set is cross-evaluated on the other half and on the held-out third block using the fixed scale 2.0 transfer intervention.

Selection overlap: `16/16`; Jaccard `1`.

| selection source | eval split | cross? | dummy-BOS | PPL ratio | BOS after | dummy after | layers |
|---|---|---:|---:|---:|---:|---:|---|
| A | A | False | 0.316239 | 1.00841 | 0.0164763 | 0.332716 | `3:14 4:2` |
| A | B | True | 0.321487 | 1.0082 | 0.0165322 | 0.338019 | `3:14 4:2` |
| A | test | True | 0.327462 | 1.00801 | 0.0167555 | 0.344217 | `3:14 4:2` |
| B | A | True | 0.316239 | 1.00841 | 0.0164763 | 0.332716 | `3:14 4:2` |
| B | B | False | 0.321487 | 1.0082 | 0.0165322 | 0.338019 | `3:14 4:2` |
| B | test | True | 0.327462 | 1.00801 | 0.0167555 | 0.344217 | `3:14 4:2` |

Interpretation: this is an appendix robustness check, not a full replacement for frozen Wikitext-to-Pile/MMLU transfer. Positive cross-eval dummy-minus-BOS indicates the reselection is not tied to a single train subset.
