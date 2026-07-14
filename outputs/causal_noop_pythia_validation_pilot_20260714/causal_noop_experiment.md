# Causal No-Op Experiment

Positive paired differences mean the control target was more disruptive than the sink target.

| model | dataset | head set | layer band | control | delta | mean control-sink KL | 95% CI | mean loss diff | mean logit L2 diff | mean correct-prob diff | sink less disruptive | sequences |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| pythia_1b | wikitext_test | low_sink | all | early | 0.05 | 0.000168948 | [8.3652e-05, 0.000262372] | 0.00171179 | 1.205 | -0.000610966 | 0.641 | 128.0 |
| pythia_1b | wikitext_test | low_sink | all | random | 0.05 | 7.1449e-05 | [-4.64964e-05, 0.000177194] | 0.00415143 | 0.68261 | -0.00137864 | 0.578 | 128.0 |
| pythia_1b | wikitext_test | low_sink | early | early | 0.05 | 0.000122592 | [-0.000159044, 0.000512281] | -0.0151375 | -2.13971 | 0.00104493 | 0.500 | 6.0 |
| pythia_1b | wikitext_test | low_sink | early | random | 0.05 | -2.21333e-05 | [-0.000314981, 0.000302321] | 0.00418053 | -3.6139 | 0.000982555 | 0.333 | 6.0 |
| pythia_1b | wikitext_test | low_sink | late | early | 0.05 | 0.00021589 | [8.83388e-05, 0.000355284] | 0.00530459 | 1.58446 | -0.0012211 | 0.658 | 79.0 |
| pythia_1b | wikitext_test | low_sink | late | random | 0.05 | 8.81102e-05 | [-7.07962e-05, 0.000256305] | 0.00873324 | 1.00926 | -0.00209649 | 0.608 | 79.0 |
| pythia_1b | wikitext_test | low_sink | middle | early | 0.05 | 8.91758e-05 | [3.11742e-05, 0.00015547] | -0.00253786 | 0.974557 | 0.000278927 | 0.628 | 43.0 |
| pythia_1b | wikitext_test | low_sink | middle | random | 0.05 | 5.38968e-05 | [-4.83419e-05, 0.000164033] | -0.00427039 | 0.682005 | -0.000389288 | 0.558 | 43.0 |
| pythia_1b | wikitext_test | sink_heavy | all | early | 0.05 | 0.000163932 | [4.44449e-05, 0.00031301] | 0.00318542 | 2.40934 | 0.000806787 | 0.632 | 106.0 |
| pythia_1b | wikitext_test | sink_heavy | all | random | 0.05 | 0.000188658 | [6.73698e-05, 0.000318894] | 0.00416527 | 2.11596 | -0.000581011 | 0.708 | 106.0 |
| pythia_1b | wikitext_test | sink_heavy | early | early | 0.05 | -5.85178e-05 | [-0.000255862, 0.000116886] | 0.00633296 | 2.17142 | -0.00462808 | 0.500 | 4.0 |
| pythia_1b | wikitext_test | sink_heavy | early | random | 0.05 | -0.000292641 | [-0.00137514, 0.000664796] | 0.0186781 | 0.862576 | -0.00966857 | 0.500 | 4.0 |
| pythia_1b | wikitext_test | sink_heavy | late | early | 0.05 | 0.000251633 | [6.42697e-05, 0.000490402] | 0.0018786 | 3.22256 | 0.000989541 | 0.712 | 66.0 |
| pythia_1b | wikitext_test | sink_heavy | late | random | 0.05 | 0.000265083 | [0.000113943, 0.000441996] | 0.00264987 | 2.86735 | -0.000119012 | 0.788 | 66.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | early | 0.05 | 2.7865e-05 | [-6.04063e-05, 0.000139694] | 0.00523152 | 0.944866 | 0.00107561 | 0.500 | 36.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | random | 0.05 | 0.000102024 | [-6.04416e-05, 0.000273838] | 0.00533096 | 0.877672 | -0.000418279 | 0.583 | 36.0 |

## Value-Content Validation

| model | dataset | control | value intervention | mean KL | mean loss change | mean logit L2 | sites |
|---|---|---|---|---:|---:|---:|---:|
| pythia_1b | wikitext_test | random | ordinary_value_at_sink | 0.0840678 | 0.0908307 | 114.467 | 128 |
| pythia_1b | wikitext_test | random | sink_value_at_ordinary | 0.000285861 | 0.000409687 | 4.34049 | 128 |

## Value-Content Paired Asymmetry

Positive paired differences mean ordinary value at the sink was more disruptive than sink value at the ordinary position.

| model | dataset | control | layer band | mean paired KL diff | 95% CI | ordinary-at-sink more disruptive | sequences |
|---|---|---|---|---:|---|---:|---:|
| pythia_1b | wikitext_test | random | all | 0.083782 | [0.0622681, 0.106484] | 0.992 | 128.0 |
| pythia_1b | wikitext_test | random | early | 0.0341054 | [0.0184588, 0.0527925] | 1.000 | 6.0 |
| pythia_1b | wikitext_test | random | late | 0.0927791 | [0.0632912, 0.128939] | 1.000 | 79.0 |
| pythia_1b | wikitext_test | random | middle | 0.0741839 | [0.0510842, 0.106221] | 0.977 | 43.0 |

## Secondary All-Head Intervention

| model | dataset | target | delta | mean KL | mean loss change | mean logit L2 | applied sites | skipped sites | sequences |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pythia_1b | wikitext_test | early | 0.05 | 0.00676179 | 0.00599944 | 38.3022 | 489471 | 1 | 128 |
| pythia_1b | wikitext_test | random | 0.05 | 0.00596239 | 0.00507137 | 34.6097 | 489469 | 3 | 128 |
| pythia_1b | wikitext_test | sink | 0.05 | 0.00209012 | 3.04751e-05 | 16.8465 | 432711 | 56761 | 128 |

## Secondary Paired Differences

Positive paired differences mean the all-head ordinary target was more disruptive than the all-head sink target.

| model | dataset | control | delta | mean control-sink KL | 95% CI | sink less disruptive | sequences |
|---|---|---|---:|---:|---|---:|---:|
| pythia_1b | wikitext_test | early | 0.05 | 0.00467167 | [0.00439765, 0.00495501] | 1.000 | 128.0 |
| pythia_1b | wikitext_test | random | 0.05 | 0.00387227 | [0.00369641, 0.00407134] | 1.000 | 128.0 |
