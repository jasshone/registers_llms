# Matched No-Op Sink vs Ordinary Table

Primary comparison: same sequence, query, layer, head, and injected attention mass; target is either the sink token or a matched ordinary token. Rows are averaged at the sequence level before bootstrapping.

- Delta: 0.05
- Head set: sink_heavy
- Ordinary target: random
- Bootstrap samples: 10000

Positive paired differences mean ordinary-token redirection was more disruptive than sink-token redirection.

| Model | Sink ΔNLL | Ordinary ΔNLL | Ordinary - sink ΔNLL | 95% CI | Fraction positive | Sink KL | Ordinary KL | Ordinary - sink KL | 95% CI | n seq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pythia_1b | -0.00162387 | -0.000157436 | 0.00146643 | [-0.00271421, 0.00551924] | 0.538 | 0.000373234 | 0.000538314 | 0.00016508 | [9.3217e-05, 0.000239696] | 273 |
| llama3_2_3b | 0.000794207 | 0.00125503 | 0.000460824 | [-0.00335682, 0.0042646] | 0.504 | 0.000426906 | 0.000354361 | -7.25442e-05 | [-0.000146355, 3.79185e-07] | 258 |
| mistral_7b_v0_1 | -0.00146431 | -4.20162e-05 | 0.0014223 | [-0.00114861, 0.00388974] | 0.487 | 0.000195152 | 0.000193879 | -1.27246e-06 | [-3.4427e-05, 2.92818e-05] | 187 |

Logit-change norm check:

| Model | Sink logit L2 | Ordinary logit L2 | Ordinary - sink logit L2 | 95% CI | Fraction positive |
|---|---:|---:|---:|---:|---:|
| pythia_1b | 5.48702 | 7.42828 | 1.94126 | [1.66212, 2.23334] | 0.872 |
| llama3_2_3b | 8.39064 | 9.61987 | 1.22923 | [0.733044, 1.7] | 0.725 |
| mistral_7b_v0_1 | 3.57259 | 3.63708 | 0.0644891 | [-9.77667e-05, 0.135002] | 0.556 |

Interpretation:

- The ΔNLL confidence intervals cross zero for the representative models, so this is not strong evidence that sink redirection is reliably less disruptive on next-token loss.
- KL and logit-change norm are sometimes positive, but the evidence is not uniformly strong enough to support a broad No-Op mechanism claim from this table alone.
- The narrower value-content claim remains safer unless a larger or more targeted matched run produces sequence-level CIs above zero.

Skipped site pairs were excluded only when either the sink or ordinary row could not receive the requested fixed attention mass because the target already had too much attention.
