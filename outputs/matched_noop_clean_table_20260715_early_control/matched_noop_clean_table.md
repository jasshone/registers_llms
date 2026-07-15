# Matched No-Op Sink vs Ordinary Table

Primary comparison: same sequence, query, layer, head, and injected attention mass; target is either the sink token or a matched ordinary token. Rows are averaged at the sequence level before bootstrapping.

- Delta: 0.05
- Head set: sink_heavy
- Ordinary target: early
- Bootstrap samples: 10000

Positive paired differences mean ordinary-token redirection was more disruptive than sink-token redirection.

| Model | Sink ΔNLL | Ordinary ΔNLL | Ordinary - sink ΔNLL | 95% CI | Fraction positive | Sink KL | Ordinary KL | Ordinary - sink KL | 95% CI | n seq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pythia_1b | -0.00162387 | 0.000242985 | 0.00186685 | [-0.00170513, 0.00546207] | 0.505 | 0.000373234 | 0.000521704 | 0.00014847 | [7.62156e-05, 0.000229117] | 273 |
| llama3_2_3b | 0.000794207 | 0.00032614 | -0.000468068 | [-0.00439628, 0.00339768] | 0.481 | 0.000426906 | 0.000421517 | -5.38815e-06 | [-8.654e-05, 7.55338e-05] | 258 |
| mistral_7b_v0_1 | -0.00146431 | 0.00054585 | 0.00201016 | [-0.00108224, 0.00514852] | 0.497 | 0.000195152 | 0.000215159 | 2.00071e-05 | [-1.82821e-05, 5.87106e-05] | 187 |

Logit-change norm check:

| Model | Sink logit L2 | Ordinary logit L2 | Ordinary - sink logit L2 | 95% CI | Fraction positive |
|---|---:|---:|---:|---:|---:|
| pythia_1b | 5.48702 | 7.82038 | 2.33336 | [2.03803, 2.63293] | 0.875 |
| llama3_2_3b | 8.39064 | 9.17909 | 0.788451 | [0.266967, 1.2562] | 0.655 |
| mistral_7b_v0_1 | 3.57259 | 3.63667 | 0.0640807 | [0.0063325, 0.121258] | 0.599 |

Interpretation:

- The ΔNLL confidence intervals cross zero for the representative models, so this is not strong evidence that sink redirection is reliably less disruptive on next-token loss.
- KL and logit-change norm are sometimes positive, but the evidence is not uniformly strong enough to support a broad No-Op mechanism claim from this table alone.
- The narrower value-content claim remains safer unless a larger or more targeted matched run produces sequence-level CIs above zero.

Skipped site pairs were excluded only when either the sink or ordinary row could not receive the requested fixed attention mass because the target already had too much attention.
