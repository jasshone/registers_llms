# Causal No-Op Experiment

Positive paired differences mean the control target was more disruptive than the sink target.

| model | dataset | head set | layer band | control | delta | mean control-sink KL | 95% CI | mean loss diff | mean logit L2 diff | mean correct-prob diff | sink less disruptive | sequences |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| pythia_1b | wikitext_test | low_sink | all | early | 0.02 | 1.46807e-05 | [-4.20731e-05, 6.86566e-05] | -0.00172214 | 0.304791 | -0.000650888 | 0.539 | 128.0 |
| pythia_1b | wikitext_test | low_sink | all | random | 0.02 | 4.84255e-05 | [-1.13794e-05, 0.000115359] | -0.00220366 | 0.086228 | -0.00029053 | 0.508 | 128.0 |
| pythia_1b | wikitext_test | low_sink | early | early | 0.02 | 3.16526e-05 | [-0.000122397, 0.000225068] | 0.0028945 | -0.984678 | -0.00199348 | 0.500 | 6.0 |
| pythia_1b | wikitext_test | low_sink | early | random | 0.02 | 0.000320192 | [-4.98145e-05, 0.000821402] | -0.00952791 | -1.1793 | 0.00427825 | 0.500 | 6.0 |
| pythia_1b | wikitext_test | low_sink | late | early | 0.02 | 3.22983e-05 | [-4.46652e-05, 0.000108955] | -0.00201431 | 0.52214 | -0.000598617 | 0.557 | 79.0 |
| pythia_1b | wikitext_test | low_sink | late | random | 0.02 | 3.55416e-05 | [-1.75357e-05, 9.54247e-05] | 0.0008896 | 0.236604 | -0.00100386 | 0.544 | 79.0 |
| pythia_1b | wikitext_test | low_sink | middle | early | 0.02 | -2.00548e-05 | [-0.000100175, 4.76148e-05] | -0.00182955 | 0.0854015 | -0.000559584 | 0.512 | 43.0 |
| pythia_1b | wikitext_test | low_sink | middle | random | 0.02 | 3.41748e-05 | [-9.33902e-05, 0.000183172] | -0.00686464 | -0.0134595 | 0.000382508 | 0.442 | 43.0 |
| pythia_1b | wikitext_test | sink_heavy | all | early | 0.02 | -6.52781e-06 | [-0.000117056, 8.86427e-05] | -0.00231494 | 0.63919 | 0.000674629 | 0.623 | 122.0 |
| pythia_1b | wikitext_test | sink_heavy | all | random | 0.02 | 8.52535e-06 | [-9.65295e-05, 8.75246e-05] | -0.00459682 | 0.605574 | 0.00124143 | 0.566 | 122.0 |
| pythia_1b | wikitext_test | sink_heavy | early | early | 0.02 | 0.000371611 | [-0.000255887, 0.00142095] | -0.0205421 | 1.34628 | 0.00855806 | 0.500 | 4.0 |
| pythia_1b | wikitext_test | sink_heavy | early | random | 0.02 | -0.000203789 | [-0.000290394, -0.000104868] | -0.00757529 | -0.205517 | 0.000795772 | 0.000 | 4.0 |
| pythia_1b | wikitext_test | sink_heavy | late | early | 0.02 | 6.21969e-05 | [-1.52665e-05, 0.000147296] | -0.00324419 | 0.740047 | 0.00097157 | 0.654 | 78.0 |
| pythia_1b | wikitext_test | sink_heavy | late | random | 0.02 | 9.38136e-05 | [3.5006e-05, 0.000164598] | -0.00530378 | 0.849422 | 0.00164795 | 0.654 | 78.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | early | 0.02 | -0.000178355 | [-0.000442477, 2.8632e-06] | 0.00131984 | 0.371811 | -0.000692748 | 0.575 | 40.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | random | 0.02 | -0.000136555 | [-0.000409846, 4.05323e-05] | -0.00292042 | 0.211179 | 0.000493276 | 0.450 | 40.0 |
| pythia_1b | wikitext_test | low_sink | all | early | 0.05 | 9.8714e-05 | [4.31018e-05, 0.00015272] | -0.000656227 | 1.17684 | -0.000112668 | 0.626 | 318.0 |
| pythia_1b | wikitext_test | low_sink | all | random | 0.05 | 7.63796e-05 | [2.01906e-05, 0.000135072] | 0.000984497 | 0.743539 | -0.000461272 | 0.575 | 318.0 |
| pythia_1b | wikitext_test | low_sink | early | early | 0.05 | -1.57088e-05 | [-0.000249452, 0.000270601] | -0.0212385 | -1.55826 | 0.00251984 | 0.353 | 17.0 |
| pythia_1b | wikitext_test | low_sink | early | random | 0.05 | 0.000117435 | [-0.000170715, 0.000435013] | -0.014857 | -1.91465 | 0.00339448 | 0.412 | 17.0 |
| pythia_1b | wikitext_test | low_sink | late | early | 0.05 | 0.000139267 | [7.70924e-05, 0.000202904] | 0.00191694 | 1.63512 | -0.000459822 | 0.660 | 203.0 |
| pythia_1b | wikitext_test | low_sink | late | random | 0.05 | 8.21215e-05 | [5.22386e-06, 0.000158486] | 0.00345527 | 1.03099 | -0.000707566 | 0.611 | 203.0 |
| pythia_1b | wikitext_test | low_sink | middle | early | 0.05 | 3.45603e-05 | [-7.5112e-05, 0.000130266] | -0.00241597 | 0.70201 | 0.000149778 | 0.602 | 98.0 |
| pythia_1b | wikitext_test | low_sink | middle | random | 0.05 | 5.73637e-05 | [-1.04037e-05, 0.000136427] | -0.00138552 | 0.609227 | -0.000619947 | 0.531 | 98.0 |
| pythia_1b | wikitext_test | sink_heavy | all | early | 0.05 | 0.00014847 | [7.46768e-05, 0.000230413] | 0.00186685 | 2.33336 | 0.000657837 | 0.659 | 273.0 |
| pythia_1b | wikitext_test | sink_heavy | all | random | 0.05 | 0.00016508 | [9.87937e-05, 0.000238863] | 0.00146643 | 1.94126 | 0.000381664 | 0.648 | 273.0 |
| pythia_1b | wikitext_test | sink_heavy | early | early | 0.05 | 3.07658e-05 | [-0.00010128, 0.000173904] | 0.00322521 | 0.755768 | -0.00189015 | 0.538 | 13.0 |
| pythia_1b | wikitext_test | sink_heavy | early | random | 0.05 | 9.87292e-05 | [-0.000326888, 0.000508779] | -0.00395131 | 0.4736 | -0.000133832 | 0.538 | 13.0 |
| pythia_1b | wikitext_test | sink_heavy | late | early | 0.05 | 0.00023374 | [0.000127618, 0.000352541] | 0.00174723 | 3.09012 | 0.0009197 | 0.731 | 175.0 |
| pythia_1b | wikitext_test | sink_heavy | late | random | 0.05 | 0.000224779 | [0.000134284, 0.000318879] | 0.00242745 | 2.61063 | 0.000394955 | 0.726 | 175.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | early | 0.05 | -9.08329e-06 | [-9.04259e-05, 6.66062e-05] | 0.00190538 | 1.01661 | 0.0005084 | 0.529 | 85.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | random | 0.05 | 5.23183e-05 | [-5.64455e-05, 0.000168906] | 0.000316472 | 0.787612 | 0.000433141 | 0.506 | 85.0 |
| pythia_1b | wikitext_test | low_sink | all | early | 0.1 | 0.000104019 | [-0.000115142, 0.000297679] | 0.000534163 | 2.80891 | -0.00042865 | 0.648 | 128.0 |
| pythia_1b | wikitext_test | low_sink | all | random | 0.1 | 7.84788e-05 | [-0.000134356, 0.000279477] | 0.00776617 | 2.13977 | -0.00161977 | 0.594 | 128.0 |
| pythia_1b | wikitext_test | low_sink | early | early | 0.1 | -0.000968187 | [-0.00260726, 0.000107467] | 0.0110625 | -6.18007 | -0.00545543 | 0.167 | 6.0 |
| pythia_1b | wikitext_test | low_sink | early | random | 0.1 | -0.00112104 | [-0.00263055, 0.000132868] | 0.0167624 | -6.41976 | -0.00818806 | 0.167 | 6.0 |
| pythia_1b | wikitext_test | low_sink | late | early | 0.1 | 0.000113565 | [-0.000210611, 0.000429227] | -0.000634016 | 3.74438 | -0.000814463 | 0.684 | 79.0 |
| pythia_1b | wikitext_test | low_sink | late | random | 0.1 | 0.000128154 | [-0.000205084, 0.000428226] | 0.00772127 | 2.87474 | -0.00206142 | 0.608 | 79.0 |
| pythia_1b | wikitext_test | low_sink | middle | early | 0.1 | 0.000236092 | [9.74289e-05, 0.000420321] | 0.00121128 | 2.34453 | 0.000981579 | 0.651 | 43.0 |
| pythia_1b | wikitext_test | low_sink | middle | random | 0.1 | 0.00015459 | [5.39842e-05, 0.000275072] | 0.00659337 | 1.98383 | 0.000108138 | 0.628 | 43.0 |
| pythia_1b | wikitext_test | sink_heavy | all | early | 0.1 | 0.000443592 | [0.00021269, 0.000763346] | 0.0103202 | 5.40335 | -0.000969421 | 0.741 | 85.0 |
| pythia_1b | wikitext_test | sink_heavy | all | random | 0.1 | 0.000458033 | [0.000134453, 0.000876762] | 0.000533934 | 4.45657 | -1.54007e-06 | 0.706 | 85.0 |
| pythia_1b | wikitext_test | sink_heavy | early | early | 0.1 | 0.000127832 | [-0.000517814, 0.000548983] | -0.00296185 | 6.30291 | 0.000141629 | 0.750 | 4.0 |
| pythia_1b | wikitext_test | sink_heavy | early | random | 0.1 | 0.000242 | [-0.000237499, 0.000777332] | -0.00220638 | 1.88711 | 0.00252187 | 0.750 | 4.0 |
| pythia_1b | wikitext_test | sink_heavy | late | early | 0.1 | 0.000649929 | [0.000354298, 0.00110569] | 0.00707474 | 6.73568 | -0.000618961 | 0.815 | 54.0 |
| pythia_1b | wikitext_test | sink_heavy | late | random | 0.1 | 0.000637781 | [0.000191354, 0.00118919] | -0.00275407 | 5.99553 | 0.000624817 | 0.759 | 54.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | early | 0.1 | 7.76965e-05 | [-0.000270266, 0.000396704] | 0.0187789 | 2.60543 | -0.00183494 | 0.593 | 27.0 |
| pythia_1b | wikitext_test | sink_heavy | middle | random | 0.1 | 0.000130541 | [-0.000285766, 0.000573605] | 0.00751591 | 1.75933 | -0.00162809 | 0.593 | 27.0 |
| llama3_2_3b | wikitext_test | low_sink | all | early | 0.02 | 3.2341e-06 | [-2.95094e-05, 3.49633e-05] | -0.000207713 | 0.306348 | 0.000146917 | 0.525 | 939.0 |
| llama3_2_3b | wikitext_test | low_sink | all | random | 0.02 | 7.97866e-07 | [-3.36682e-05, 3.64151e-05] | 0.000777688 | 0.27029 | -0.000106894 | 0.515 | 939.0 |
| llama3_2_3b | wikitext_test | low_sink | early | early | 0.02 | 3.30759e-05 | [-3.15434e-05, 9.76907e-05] | 0.00071027 | 0.108373 | 0.00048359 | 0.546 | 282.0 |
| llama3_2_3b | wikitext_test | low_sink | early | random | 0.02 | -2.63417e-06 | [-7.1991e-05, 7.19727e-05] | 0.0029668 | 0.0575415 | -0.000185358 | 0.479 | 282.0 |
| llama3_2_3b | wikitext_test | low_sink | late | early | 0.02 | -9.57471e-06 | [-4.62853e-05, 2.77124e-05] | -0.000601733 | 0.391323 | 2.4093e-06 | 0.516 | 657.0 |
| llama3_2_3b | wikitext_test | low_sink | late | random | 0.02 | 2.27098e-06 | [-3.72816e-05, 3.79913e-05] | -0.000161932 | 0.361607 | -7.32149e-05 | 0.531 | 657.0 |
| llama3_2_3b | wikitext_test | sink_heavy | all | early | 0.02 | 1.12957e-05 | [-3.17374e-05, 5.07649e-05] | 0.000171008 | 0.16424 | -1.27407e-05 | 0.503 | 557.0 |
| llama3_2_3b | wikitext_test | sink_heavy | all | random | 0.02 | 6.87365e-07 | [-4.56885e-05, 4.7087e-05] | 0.0013347 | 0.338649 | -0.000262012 | 0.503 | 557.0 |
| llama3_2_3b | wikitext_test | sink_heavy | early | early | 0.02 | 2.94638e-05 | [-9.3799e-05, 0.000155706] | -0.00335291 | 0.134966 | 0.000582787 | 0.466 | 103.0 |
| llama3_2_3b | wikitext_test | sink_heavy | early | random | 0.02 | 6.32885e-05 | [-6.58095e-05, 0.000199087] | -0.00316162 | 0.287579 | -0.000208348 | 0.524 | 103.0 |
| llama3_2_3b | wikitext_test | sink_heavy | late | early | 0.02 | 7.17386e-06 | [-3.59521e-05, 4.71071e-05] | 0.000970488 | 0.170881 | -0.000147849 | 0.511 | 454.0 |
| llama3_2_3b | wikitext_test | sink_heavy | late | random | 0.02 | -1.35151e-05 | [-6.1157e-05, 3.42836e-05] | 0.00235479 | 0.350235 | -0.000274187 | 0.498 | 454.0 |
| llama3_2_3b | wikitext_test | low_sink | all | early | 0.05 | -1.04187e-06 | [-3.66513e-05, 3.22686e-05] | 0.000276806 | 0.886718 | -0.00013434 | 0.561 | 1108.0 |
| llama3_2_3b | wikitext_test | low_sink | all | random | 0.05 | 2.59705e-05 | [-9.86727e-06, 6.13179e-05] | 0.00118017 | 0.647939 | -0.000267314 | 0.560 | 1108.0 |
| llama3_2_3b | wikitext_test | low_sink | early | early | 0.05 | 1.92517e-05 | [-4.58871e-05, 8.26067e-05] | 0.000523391 | 0.229458 | 0.000242962 | 0.548 | 336.0 |
| llama3_2_3b | wikitext_test | low_sink | early | random | 0.05 | 5.80538e-06 | [-6.47219e-05, 7.06622e-05] | 0.001561 | 0.332787 | 0.000427403 | 0.533 | 336.0 |
| llama3_2_3b | wikitext_test | low_sink | late | early | 0.05 | -9.87433e-06 | [-5.11775e-05, 3.15317e-05] | 0.000169484 | 1.17278 | -0.000298554 | 0.567 | 772.0 |
| llama3_2_3b | wikitext_test | low_sink | late | random | 0.05 | 3.4747e-05 | [-4.98003e-06, 7.57634e-05] | 0.00101443 | 0.785104 | -0.000569678 | 0.573 | 772.0 |
| llama3_2_3b | wikitext_test | sink_heavy | all | early | 0.05 | -5.38815e-06 | [-8.49672e-05, 7.71558e-05] | -0.000468068 | 0.788451 | 5.01597e-05 | 0.508 | 258.0 |
| llama3_2_3b | wikitext_test | sink_heavy | all | random | 0.05 | -7.25442e-05 | [-0.000144928, 1.16216e-06] | 0.000460824 | 1.22923 | -1.93388e-05 | 0.535 | 258.0 |
| llama3_2_3b | wikitext_test | sink_heavy | early | early | 0.05 | -1.09745e-05 | [-0.000283377, 0.000299119] | 0.00363059 | -0.0863149 | -0.00220645 | 0.487 | 39.0 |
| llama3_2_3b | wikitext_test | sink_heavy | early | random | 0.05 | -0.000295832 | [-0.000544851, -5.78668e-05] | 0.00127657 | 0.60505 | -0.00133667 | 0.333 | 39.0 |
| llama3_2_3b | wikitext_test | sink_heavy | late | early | 0.05 | -4.39333e-06 | [-8.82868e-05, 7.68274e-05] | -0.00119797 | 0.944231 | 0.000452022 | 0.511 | 219.0 |
| llama3_2_3b | wikitext_test | sink_heavy | late | random | 0.05 | -3.27806e-05 | [-0.00010702, 4.12065e-05] | 0.000315553 | 1.34039 | 0.000215254 | 0.571 | 219.0 |
| llama3_2_3b | wikitext_test | low_sink | all | early | 0.1 | 9.53779e-05 | [5.39765e-05, 0.000134524] | -0.00122528 | 2.27868 | 0.000174417 | 0.584 | 877.0 |
| llama3_2_3b | wikitext_test | low_sink | all | random | 0.1 | 7.09014e-05 | [1.9593e-05, 0.000124912] | -0.000645724 | 1.66593 | 0.000210533 | 0.574 | 877.0 |
| llama3_2_3b | wikitext_test | low_sink | early | early | 0.1 | 9.29614e-05 | [1.3353e-05, 0.000175269] | 0.000867043 | 1.15906 | 0.000416514 | 0.532 | 282.0 |
| llama3_2_3b | wikitext_test | low_sink | early | random | 0.1 | 5.67431e-05 | [-2.78847e-05, 0.000145957] | 0.000679509 | 1.25947 | 0.000202685 | 0.557 | 282.0 |
| llama3_2_3b | wikitext_test | low_sink | late | early | 0.1 | 9.65232e-05 | [5.10773e-05, 0.000142691] | -0.00221694 | 2.80933 | 5.96756e-05 | 0.608 | 595.0 |
| llama3_2_3b | wikitext_test | low_sink | late | random | 0.1 | 7.76117e-05 | [1.94796e-05, 0.000143149] | -0.00127382 | 1.85856 | 0.000214252 | 0.582 | 595.0 |
| llama3_2_3b | wikitext_test | sink_heavy | all | early | 0.1 | -5.05518e-06 | [-0.000184723, 0.000178979] | -0.00442617 | 1.29649 | 0.00101705 | 0.644 | 73.0 |
| llama3_2_3b | wikitext_test | sink_heavy | all | random | 0.1 | -1.87491e-05 | [-0.000154452, 0.000108059] | 0.00128689 | 1.66376 | 0.000656829 | 0.562 | 73.0 |
| llama3_2_3b | wikitext_test | sink_heavy | early | early | 0.1 | -0.000213651 | [-0.000628967, 0.000120644] | -0.00979679 | 0.94301 | 0.00368765 | 0.800 | 10.0 |
| llama3_2_3b | wikitext_test | sink_heavy | early | random | 0.1 | -0.000166129 | [-0.000566685, 0.000155504] | -0.00077549 | -0.53236 | 0.00177857 | 0.500 | 10.0 |
| llama3_2_3b | wikitext_test | sink_heavy | late | early | 0.1 | 2.80553e-05 | [-0.000164135, 0.000229165] | -0.00357369 | 1.3526 | 0.000593144 | 0.619 | 63.0 |
| llama3_2_3b | wikitext_test | sink_heavy | late | random | 0.1 | 4.64455e-06 | [-0.000142596, 0.000151891] | 0.00161425 | 2.01235 | 0.000478774 | 0.571 | 63.0 |
| qwen3_4b | wikitext_test | low_sink | all | early | 0.02 | -1.8499e-05 | [-0.000102781, 6.53523e-05] | -2.2343e-05 | 0.241605 | -6.88832e-06 | 0.475 | 977.0 |
| qwen3_4b | wikitext_test | low_sink | all | random | 0.02 | -2.75754e-05 | [-9.27948e-05, 3.70208e-05] | 0.00220196 | -0.13257 | -0.000185478 | 0.493 | 977.0 |
| qwen3_4b | wikitext_test | low_sink | early | early | 0.02 | -9.63931e-05 | [-0.000271928, 6.14091e-05] | -0.00181139 | -0.642937 | 0.000374351 | 0.466 | 307.0 |
| qwen3_4b | wikitext_test | low_sink | early | random | 0.02 | -4.26708e-05 | [-0.000173477, 8.95088e-05] | 0.00389035 | -0.740456 | -0.000342923 | 0.469 | 307.0 |
| qwen3_4b | wikitext_test | low_sink | late | early | 0.02 | 5.4439e-05 | [-3.46376e-05, 0.000163534] | 0.00054411 | 0.152966 | -0.000273139 | 0.481 | 539.0 |
| qwen3_4b | wikitext_test | low_sink | late | random | 0.02 | 1.76363e-05 | [-4.7661e-05, 8.59026e-05] | -0.000296949 | -0.0743163 | 0.000111869 | 0.508 | 539.0 |
| qwen3_4b | wikitext_test | low_sink | middle | early | 0.02 | -0.000136057 | [-0.000347852, 8.7483e-05] | 0.00183963 | 2.67925 | 0.000195161 | 0.473 | 131.0 |
| qwen3_4b | wikitext_test | low_sink | middle | random | 0.02 | -0.000178223 | [-0.000445991, 4.27623e-05] | 0.00852697 | 1.05233 | -0.00103994 | 0.489 | 131.0 |
| qwen3_4b | wikitext_test | sink_heavy | all | early | 0.02 | 7.40031e-05 | [-3.60245e-06, 0.000150176] | 0.00158247 | -0.0209969 | -0.000126503 | 0.549 | 379.0 |
| qwen3_4b | wikitext_test | sink_heavy | all | random | 0.02 | 1.8076e-05 | [-5.99571e-05, 9.38307e-05] | -0.00164766 | 0.535974 | 0.000158881 | 0.557 | 379.0 |
| qwen3_4b | wikitext_test | sink_heavy | early | early | 0.02 | -6.2036e-05 | [-0.000285595, 0.00018615] | 0.0129177 | -0.221447 | -0.00218833 | 0.412 | 68.0 |
| qwen3_4b | wikitext_test | sink_heavy | early | random | 0.02 | -4.83222e-05 | [-0.000309254, 0.000220673] | 0.00769144 | -0.599357 | 0.000535616 | 0.456 | 68.0 |
| qwen3_4b | wikitext_test | sink_heavy | late | early | 0.02 | 9.79123e-05 | [1.75928e-05, 0.000180267] | -0.00154384 | 0.216047 | 0.000218776 | 0.573 | 286.0 |
| qwen3_4b | wikitext_test | sink_heavy | late | random | 0.02 | 2.81152e-05 | [-4.82401e-05, 0.000101721] | -0.00425895 | 0.877898 | 1.94344e-05 | 0.587 | 286.0 |
| qwen3_4b | wikitext_test | sink_heavy | middle | early | 0.02 | 0.000170508 | [-2.69109e-05, 0.00036416] | 0.00651567 | -2.18756 | 0.00153168 | 0.640 | 25.0 |
| qwen3_4b | wikitext_test | sink_heavy | middle | random | 0.02 | 8.38305e-05 | [-0.000217744, 0.000390722] | 0.00282312 | -0.287543 | 0.000729427 | 0.480 | 25.0 |
| qwen3_4b | wikitext_test | low_sink | all | early | 0.05 | -5.55171e-05 | [-0.000112618, 2.27177e-07] | 0.00342201 | 0.438647 | -0.000302802 | 0.491 | 1158.0 |
| qwen3_4b | wikitext_test | low_sink | all | random | 0.05 | 1.23486e-05 | [-4.76603e-05, 7.77078e-05] | 0.00257588 | 0.505783 | 2.19163e-05 | 0.490 | 1158.0 |
| qwen3_4b | wikitext_test | low_sink | early | early | 0.05 | -0.000120179 | [-0.000242821, -1.07311e-05] | 0.001206 | -0.105435 | 6.9917e-05 | 0.491 | 375.0 |
| qwen3_4b | wikitext_test | low_sink | early | random | 0.05 | 2.16855e-05 | [-9.29806e-05, 0.000137381] | 0.00697599 | 0.299658 | -0.00035034 | 0.496 | 375.0 |
| qwen3_4b | wikitext_test | low_sink | late | early | 0.05 | -3.46064e-05 | [-0.000101924, 3.48052e-05] | 0.00561246 | 0.575177 | -0.000533395 | 0.496 | 623.0 |
| qwen3_4b | wikitext_test | low_sink | late | random | 0.05 | -3.90928e-06 | [-7.84734e-05, 7.15349e-05] | 0.000571171 | 1.09183 | 0.000521246 | 0.491 | 623.0 |
| qwen3_4b | wikitext_test | low_sink | middle | early | 0.05 | 1.46142e-05 | [-0.000126161, 0.000166227] | 8.67306e-05 | 1.18222 | -0.000278491 | 0.475 | 160.0 |
| qwen3_4b | wikitext_test | low_sink | middle | random | 0.05 | 5.37696e-05 | [-0.000174964, 0.000269763] | 6.89513e-05 | -1.29303 | -0.00104987 | 0.469 | 160.0 |
| qwen3_4b | wikitext_test | sink_heavy | all | early | 0.05 | 2.68225e-05 | [-9.74436e-05, 0.00014462] | 0.00200594 | 0.685411 | 0.000801448 | 0.503 | 187.0 |
| qwen3_4b | wikitext_test | sink_heavy | all | random | 0.05 | 4.6619e-06 | [-0.000118818, 0.000123762] | 0.00272296 | 1.42071 | 0.000296992 | 0.487 | 187.0 |
| qwen3_4b | wikitext_test | sink_heavy | early | early | 0.05 | -0.000238665 | [-0.000460846, -3.02136e-05] | 0.00881305 | -1.06346 | 0.000139662 | 0.353 | 34.0 |
| qwen3_4b | wikitext_test | sink_heavy | early | random | 0.05 | -0.000187515 | [-0.000419687, 3.98948e-05] | 0.011268 | 1.31305 | -0.00050318 | 0.324 | 34.0 |
| qwen3_4b | wikitext_test | sink_heavy | late | early | 0.05 | 0.000112544 | [1.07137e-06, 0.000229886] | -0.000988823 | 1.55639 | 0.000641254 | 0.533 | 135.0 |
| qwen3_4b | wikitext_test | sink_heavy | late | random | 0.05 | 8.75935e-05 | [-2.27121e-05, 0.000200666] | -0.00131422 | 1.88166 | -5.84007e-05 | 0.533 | 135.0 |
| qwen3_4b | wikitext_test | sink_heavy | middle | early | 0.05 | -0.000114612 | [-0.00100795, 0.000604846] | 0.0116088 | -2.54355 | 0.00325294 | 0.556 | 18.0 |
| qwen3_4b | wikitext_test | sink_heavy | middle | random | 0.05 | -0.000254324 | [-0.00102673, 0.000432914] | 0.0168611 | -1.83307 | 0.00447387 | 0.444 | 18.0 |
| qwen3_4b | wikitext_test | low_sink | all | early | 0.1 | 7.22357e-05 | [3.54108e-06, 0.000142268] | 0.00346351 | 1.18282 | -0.000654416 | 0.536 | 957.0 |
| qwen3_4b | wikitext_test | low_sink | all | random | 0.1 | 4.20345e-05 | [-2.93887e-05, 0.000115531] | 0.00233677 | 0.541313 | -0.000102025 | 0.507 | 957.0 |
| qwen3_4b | wikitext_test | low_sink | early | early | 0.1 | 0.000111577 | [-7.80407e-06, 0.000238546] | 0.00152955 | -0.368487 | -0.00018724 | 0.536 | 306.0 |
| qwen3_4b | wikitext_test | low_sink | early | random | 0.1 | 0.000107502 | [-2.22995e-05, 0.000244605] | 0.00203906 | -0.0510392 | -0.000334164 | 0.510 | 306.0 |
| qwen3_4b | wikitext_test | low_sink | late | early | 0.1 | 2.79123e-05 | [-5.40522e-05, 0.000121098] | 0.00225397 | 1.0688 | -0.000715995 | 0.533 | 520.0 |
| qwen3_4b | wikitext_test | low_sink | late | random | 0.1 | 6.51e-05 | [-3.2539e-05, 0.000159012] | 0.000258587 | 0.922865 | 0.000211383 | 0.525 | 520.0 |
| qwen3_4b | wikitext_test | low_sink | middle | early | 0.1 | 0.00015628 | [-8.47728e-05, 0.00043137] | 0.0127823 | 5.25912 | -0.00150125 | 0.550 | 131.0 |
| qwen3_4b | wikitext_test | low_sink | middle | random | 0.1 | -0.000202448 | [-0.000402287, -2.34682e-05] | 0.0112815 | 0.410418 | -0.000803838 | 0.427 | 131.0 |
| qwen3_4b | wikitext_test | sink_heavy | all | early | 0.1 | -1.47553e-05 | [-0.000227238, 0.000231508] | -0.00162154 | 1.75199 | 0.000210666 | 0.492 | 63.0 |
| qwen3_4b | wikitext_test | sink_heavy | all | random | 0.1 | 0.000107681 | [-0.000182433, 0.000410955] | -0.00996052 | 3.18207 | 0.00185238 | 0.571 | 63.0 |
| qwen3_4b | wikitext_test | sink_heavy | early | early | 0.1 | -1.90666e-05 | [-0.000183845, 0.000144351] | -0.0148758 | 2.09191 | 0.0013201 | 0.583 | 12.0 |
| qwen3_4b | wikitext_test | sink_heavy | early | random | 0.1 | 0.000242501 | [-4.37445e-06, 0.000538644] | -0.0263133 | 3.22684 | 0.00171724 | 0.583 | 12.0 |
| qwen3_4b | wikitext_test | sink_heavy | late | early | 0.1 | -0.000124715 | [-0.000367247, 0.000109398] | 0.00307586 | 1.46588 | 0.000529858 | 0.455 | 44.0 |
| qwen3_4b | wikitext_test | sink_heavy | late | random | 0.1 | 0.000106598 | [-0.000270444, 0.00049181] | -0.00122797 | 3.04823 | 0.00163801 | 0.568 | 44.0 |
| qwen3_4b | wikitext_test | sink_heavy | middle | early | 0.1 | 0.000683809 | [1.06335e-06, 0.00192533] | -0.00842659 | 2.96766 | -0.00369757 | 0.571 | 7.0 |
| qwen3_4b | wikitext_test | sink_heavy | middle | random | 0.1 | -0.000116629 | [-0.000615936, 0.000367498] | -0.0368175 | 3.94663 | 0.00343149 | 0.571 | 7.0 |

## Value-Content Validation

| model | dataset | control | value intervention | mean KL | mean loss change | mean logit L2 | sites |
|---|---|---|---|---:|---:|---:|---:|
| pythia_1b | wikitext_test | random | ordinary_value_at_sink | 0.0840678 | 0.0908307 | 114.467 | 128 |
| pythia_1b | wikitext_test | random | sink_value_at_ordinary | 0.000285861 | 0.000409687 | 4.34049 | 128 |
| llama3_2_3b | wikitext_test | random | ordinary_value_at_sink | 0.672239 | 0.648945 | 302.589 | 128 |
| llama3_2_3b | wikitext_test | random | sink_value_at_ordinary | 0.00034279 | -0.00273348 | 6.87973 | 128 |
| qwen3_4b | wikitext_test | random | ordinary_value_at_sink | 0.0270316 | 0.0179416 | 118.387 | 128 |
| qwen3_4b | wikitext_test | random | sink_value_at_ordinary | 0.000818522 | -0.00244944 | 16.5393 | 128 |

## Value-Content Paired Asymmetry

Positive paired differences mean ordinary value at the sink was more disruptive than sink value at the ordinary position.

| model | dataset | control | layer band | mean paired KL diff | 95% CI | ordinary-at-sink more disruptive | sequences |
|---|---|---|---|---:|---|---:|---:|
| pythia_1b | wikitext_test | random | all | 0.083782 | [0.0622681, 0.106484] | 0.992 | 128.0 |
| pythia_1b | wikitext_test | random | early | 0.0341054 | [0.0184588, 0.0527925] | 1.000 | 6.0 |
| pythia_1b | wikitext_test | random | late | 0.0927791 | [0.0632912, 0.128939] | 1.000 | 79.0 |
| pythia_1b | wikitext_test | random | middle | 0.0741839 | [0.0510842, 0.106221] | 0.977 | 43.0 |
| llama3_2_3b | wikitext_test | random | all | 0.671896 | [0.421045, 0.950387] | 1.000 | 128.0 |
| llama3_2_3b | wikitext_test | random | early | 1.83562 | [1.2599, 2.47688] | 1.000 | 46.0 |
| llama3_2_3b | wikitext_test | random | late | 0.0190742 | [0.0134276, 0.0251473] | 1.000 | 82.0 |
| qwen3_4b | wikitext_test | random | all | 0.0262131 | [0.0171032, 0.0377856] | 0.961 | 128.0 |
| qwen3_4b | wikitext_test | random | early | 0.0360616 | [0.0173283, 0.0579504] | 0.952 | 42.0 |
| qwen3_4b | wikitext_test | random | late | 0.0221818 | [0.0110778, 0.0385927] | 0.986 | 69.0 |
| qwen3_4b | wikitext_test | random | middle | 0.0182437 | [0.00388773, 0.043939] | 0.882 | 17.0 |

## Secondary All-Head Intervention

| model | dataset | target | delta | mean KL | mean loss change | mean logit L2 | applied sites | skipped sites | sequences |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pythia_1b | wikitext_test | early | 0.05 | 0.00676179 | 0.00599944 | 38.3022 | 489471 | 1 | 128 |
| pythia_1b | wikitext_test | random | 0.05 | 0.00596239 | 0.00507137 | 34.6097 | 489469 | 3 | 128 |
| pythia_1b | wikitext_test | sink | 0.05 | 0.00209012 | 3.04751e-05 | 16.8465 | 432711 | 56761 | 128 |
| llama3_2_3b | wikitext_test | early | 0.05 | 0.00266022 | 0.00134642 | 33.904 | 489472 | 0 | 128 |
| llama3_2_3b | wikitext_test | random | 0.05 | 0.00270175 | 0.000923249 | 34.7465 | 489472 | 0 | 128 |
| llama3_2_3b | wikitext_test | sink | 0.05 | 0.00124267 | 0.000196767 | 18.6539 | 115338 | 374134 | 128 |
| qwen3_4b | wikitext_test | early | 0.05 | 0.00248963 | 0.00690918 | 37.3079 | 489472 | 0 | 128 |
| qwen3_4b | wikitext_test | random | 0.05 | 0.00215729 | 0.00455867 | 36.0511 | 489472 | 0 | 128 |
| qwen3_4b | wikitext_test | sink | 0.05 | 0.00138092 | 0.000228898 | 26.1961 | 87626 | 401846 | 128 |

## Secondary Paired Differences

Positive paired differences mean the all-head ordinary target was more disruptive than the all-head sink target.

| model | dataset | control | delta | mean control-sink KL | 95% CI | sink less disruptive | sequences |
|---|---|---|---:|---:|---|---:|---:|
| pythia_1b | wikitext_test | early | 0.05 | 0.00467167 | [0.00439765, 0.00495501] | 1.000 | 128.0 |
| pythia_1b | wikitext_test | random | 0.05 | 0.00387227 | [0.00369641, 0.00407134] | 1.000 | 128.0 |
| llama3_2_3b | wikitext_test | early | 0.05 | 0.00141754 | [0.00133181, 0.00150258] | 0.992 | 128.0 |
| llama3_2_3b | wikitext_test | random | 0.05 | 0.00145907 | [0.0013539, 0.00158558] | 1.000 | 128.0 |
| qwen3_4b | wikitext_test | early | 0.05 | 0.00110872 | [0.0010087, 0.00122669] | 1.000 | 128.0 |
| qwen3_4b | wikitext_test | random | 0.05 | 0.000776378 | [0.000712266, 0.000842183] | 0.984 | 128.0 |
