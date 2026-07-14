# Causal No-Op Experiment

Positive paired differences mean the control target was more disruptive than the sink target.

| model | dataset | head set | layer band | control | delta | mean control-sink KL | 95% CI | mean loss diff | mean logit L2 diff | mean correct-prob diff | sink less disruptive | sequences |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| pythia_1b | wikitext_test | low_sink | all | early | 0.05 | 0.000120005 | [5.34874e-05, 0.000184885] | -0.000321365 | 1.16421 | -0.000424983 | 0.613 | 191.0 |
| pythia_1b | wikitext_test | low_sink | all | random | 0.05 | 8.93994e-05 | [6.04298e-06, 0.000169809] | 0.00324935 | 0.785907 | -0.00120578 | 0.607 | 191.0 |
| pythia_1b | wikitext_test | low_sink | early | early | 0.05 | -3.49279e-05 | [-0.00025234, 0.000253033] | -0.0258252 | -1.80929 | 0.00201122 | 0.300 | 10.0 |
| pythia_1b | wikitext_test | low_sink | early | random | 0.05 | 5.16114e-05 | [-0.000208938, 0.000367283] | -0.010533 | -2.54132 | 0.00130728 | 0.300 | 10.0 |
| pythia_1b | wikitext_test | low_sink | late | early | 0.05 | 0.000172033 | [7.98102e-05, 0.000272848] | 0.00258654 | 1.63714 | -0.000901871 | 0.655 | 119.0 |
| pythia_1b | wikitext_test | low_sink | late | random | 0.05 | 9.8455e-05 | [-1.24497e-05, 0.000208639] | 0.00600182 | 1.17711 | -0.00140429 | 0.647 | 119.0 |
| pythia_1b | wikitext_test | low_sink | middle | early | 0.05 | 4.51329e-05 | [-3.86811e-05, 0.000121113] | -0.00178915 | 0.736086 | 9.73968e-05 | 0.581 | 62.0 |
| pythia_1b | wikitext_test | low_sink | middle | random | 0.05 | 7.81133e-05 | [-1.39481e-05, 0.00018069] | 0.000189352 | 0.571705 | -0.0012301 | 0.581 | 62.0 |
| pythia_1b | wikitext_test | sink_heavy | all | early | 0.05 | 0.000156888 | [5.36554e-05, 0.00027386] | 0.00111451 | 2.31909 | 0.00090593 | 0.634 | 164.0 |
| pythia_1b | wikitext_test | sink_heavy | all | random | 0.05 | 0.000148348 | [5.42972e-05, 0.000248444] | 0.00266103 | 1.9535 | -0.000293358 | 0.671 | 164.0 |
| pythia_1b | wikitext_test | sink_heavy | early | early | 0.05 | 4.06432e-05 | [-0.00016576, 0.000258892] | 0.0166934 | 1.59308 | -0.00452826 | 0.571 | 7.0 |
| pythia_1b | wikitext_test | sink_heavy | early | random | 0.05 | -0.000146 | [-0.000839185, 0.000407764] | 0.00202491 | 0.817615 | -0.00589358 | 0.571 | 7.0 |
| pythia_1b | wikitext_test | sink_heavy | late | early | 0.05 | 0.000243513 | [9.14958e-05, 0.000411086] | -0.000640779 | 3.10309 | 0.00135749 | 0.709 | 103.0 |
| pythia_1b | wikitext_test | sink_heavy | late | random | 0.05 | 0.000198149 | [7.72435e-05, 0.000323709] | 0.00414558 | 2.66121 | -0.000192281 | 0.757 | 103.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | early | 0.05 | 6.72684e-06 | [-8.56657e-05, 9.7431e-05] | 0.00244307 | 0.917808 | 0.000749051 | 0.500 | 54.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | random | 0.05 | 9.15148e-05 | [-5.15562e-05, 0.000262107] | -8.81615e-05 | 0.750862 | 0.000239804 | 0.519 | 54.0 |
