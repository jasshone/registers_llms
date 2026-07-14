# Causal No-Op Experiment

Positive paired differences mean the control target was more disruptive than the sink target.

| model | dataset | head set | layer band | control | delta | mean control-sink KL | 95% CI | mean loss diff | mean logit L2 diff | mean correct-prob diff | sink less disruptive | sequences |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| mistral_7b_v0_1 | wikitext_test | low_sink | all | early | 0.02 | 1.84376e-05 | [-2.98082e-05, 7.26278e-05] | -0.00340481 | 0.0536319 | 0.000190243 | 0.547 | 64.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | all | random | 0.02 | 6.69245e-05 | [1.44374e-05, 0.000121851] | -0.00306341 | 0.166812 | 0.000397725 | 0.594 | 64.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | early | early | 0.02 | 3.72834e-05 | [-3.56124e-05, 0.00011973] | -0.0047021 | 0.104633 | 6.84953e-05 | 0.529 | 34.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | early | random | 0.02 | 6.40255e-05 | [-1.21142e-05, 0.000147365] | -0.000716176 | 0.23411 | -0.000344682 | 0.676 | 34.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | late | early | 0.02 | -1.25412e-05 | [-5.06377e-05, 2.56639e-05] | 0.00110335 | 0.0299114 | 0.000583578 | 0.500 | 22.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | late | random | 0.02 | 1.95709e-05 | [-1.63602e-05, 6.68838e-05] | -0.0028564 | 0.142171 | 0.00128025 | 0.409 | 22.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | middle | early | 0.02 | 2.35351e-05 | [-0.000148376, 0.000186705] | -0.0102888 | -0.0978913 | -0.000374002 | 0.750 | 8.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | middle | random | 0.02 | 0.000209468 | [8.8434e-06, 0.000447011] | -0.0136084 | -0.0514397 | 0.00112602 | 0.750 | 8.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | all | early | 0.02 | 1.1934e-05 | [-3.0635e-05, 5.84342e-05] | 0.00196458 | 0.0211786 | -0.000759265 | 0.522 | 46.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | all | random | 0.02 | -1.85069e-06 | [-4.47609e-05, 4.08741e-05] | -0.0035449 | -0.0463089 | 0.000583155 | 0.478 | 46.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | early | early | 0.02 | 4.38545e-06 | [-5.24048e-05, 6.1439e-05] | 0.0019524 | -0.020734 | -0.00114103 | 0.469 | 32.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | early | random | 0.02 | 1.59536e-06 | [-5.39002e-05, 5.59788e-05] | -0.00629167 | -0.0919153 | 0.000529391 | 0.500 | 32.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | late | early | 0.02 | 4.82215e-05 | [-3.93459e-06, 0.000134793] | 0.0016154 | 0.187378 | 0.000114658 | 0.727 | 11.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | late | random | 0.02 | 1.53868e-05 | [-2.42005e-05, 6.31245e-05] | -8.14631e-05 | 0.0595716 | 7.22576e-05 | 0.364 | 11.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | middle | early | 0.02 | -4.06029e-05 | [-0.000113323, 2.5583e-05] | 0.00337489 | -0.141152 | 0.000108505 | 0.333 | 3.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | middle | random | 0.02 | -0.000101813 | [-0.000396958, 5.9434e-05] | 0.0130547 | 0.0519307 | 0.00302992 | 0.667 | 3.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | all | early | 0.05 | 1.32345e-05 | [-1.10239e-05, 3.75774e-05] | -0.000168278 | 0.0606557 | 0.000137787 | 0.522 | 251.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | all | random | 0.05 | 6.54911e-06 | [-2.56434e-05, 3.74946e-05] | 0.000106538 | 0.0635814 | 0.000453462 | 0.510 | 251.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | early | early | 0.05 | 2.44563e-05 | [-1.34227e-05, 6.57867e-05] | -0.00123527 | 0.0859774 | 0.000233812 | 0.503 | 143.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | early | random | 0.05 | -5.1882e-06 | [-4.60019e-05, 3.38254e-05] | -0.00198146 | 0.096111 | 0.00056294 | 0.483 | 143.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | late | early | 0.05 | -1.37865e-05 | [-4.98913e-05, 2.31862e-05] | 0.00137159 | 0.0697296 | 0.000108245 | 0.545 | 88.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | late | random | 0.05 | 1.94156e-05 | [-3.5568e-05, 8.84902e-05] | 0.00305507 | 0.0433765 | 0.000310279 | 0.545 | 88.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | middle | early | 0.05 | 5.18901e-05 | [-1.11169e-05, 0.000119262] | 0.000685283 | -0.16032 | -0.000418805 | 0.550 | 20.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | middle | random | 0.05 | 3.38583e-05 | [-6.73078e-05, 0.000145018] | 0.00206215 | -0.0801033 | 0.000300704 | 0.550 | 20.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | all | early | 0.05 | 2.00071e-05 | [-1.87738e-05, 5.93161e-05] | 0.00201016 | 0.0640807 | -0.000508373 | 0.508 | 187.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | all | random | 0.05 | -1.27246e-06 | [-3.49262e-05, 2.99468e-05] | 0.0014223 | 0.0644891 | -0.00012315 | 0.513 | 187.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | early | early | 0.05 | -3.1089e-06 | [-4.92769e-05, 4.05466e-05] | 0.0017929 | 0.00221271 | -0.0008085 | 0.479 | 121.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | early | random | 0.05 | -1.6861e-05 | [-6.23522e-05, 2.18223e-05] | 0.00251892 | 0.0347873 | -0.000406643 | 0.446 | 121.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | late | early | 0.05 | 7.31624e-05 | [-3.84481e-06, 0.000165108] | 0.00462095 | 0.222621 | -0.000968238 | 0.585 | 53.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | late | random | 0.05 | 2.03937e-05 | [-2.69186e-05, 6.92352e-05] | 0.000956976 | 0.13204 | 0.000306702 | 0.623 | 53.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | middle | early | 0.05 | 1.84538e-05 | [-0.000110926, 0.000144438] | -0.00661158 | -0.00642547 | 0.00415995 | 0.462 | 13.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | middle | random | 0.05 | 5.54897e-05 | [-7.77278e-05, 0.000169855] | -0.00688767 | 0.0655441 | 0.00076303 | 0.692 | 13.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | all | early | 0.1 | 7.34809e-05 | [-2.27268e-05, 0.000184841] | -0.00223896 | 0.580475 | 0.000784848 | 0.574 | 61.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | all | random | 0.1 | 7.87902e-05 | [-8.86715e-06, 0.00018132] | 0.00079218 | 0.427242 | -7.91432e-05 | 0.607 | 61.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | early | early | 0.1 | 7.88475e-05 | [-7.37532e-05, 0.000271064] | -0.00601451 | 0.827072 | 0.00022997 | 0.484 | 31.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | early | random | 0.1 | 9.0805e-05 | [-6.64459e-05, 0.000283352] | -0.00397128 | 0.579398 | -0.000416868 | 0.613 | 31.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | late | early | 0.1 | 0.000111943 | [1.5686e-05, 0.000264437] | 0.000878523 | 0.36144 | 0.000923847 | 0.727 | 22.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | late | random | 0.1 | 4.60552e-05 | [-2.84505e-06, 0.000106184] | 0.00240859 | 0.26783 | 0.000260112 | 0.545 | 22.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | middle | early | 0.1 | -5.3086e-05 | [-0.000301773, 0.000138489] | 0.00381818 | 0.227257 | 0.00255275 | 0.500 | 8.0 |
| mistral_7b_v0_1 | wikitext_test | low_sink | middle | random | 0.1 | 0.000122254 | [-6.23521e-05, 0.000317541] | 0.0148055 | 0.276017 | 0.00029659 | 0.750 | 8.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | all | early | 0.1 | 5.60996e-05 | [-8.979e-06, 0.000135392] | 0.000760591 | 0.0697548 | -0.00072773 | 0.654 | 26.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | all | random | 0.1 | 7.99554e-05 | [4.72726e-07, 0.000167233] | -0.00158439 | 0.00658424 | -0.000740801 | 0.731 | 26.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | early | early | 0.1 | 4.17874e-05 | [-5.95549e-05, 0.00015873] | 0.00395641 | -0.206344 | -0.00155732 | 0.600 | 15.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | early | random | 0.1 | 7.94704e-05 | [-4.65751e-05, 0.000226998] | 0.00458814 | -0.280073 | -0.0014858 | 0.600 | 15.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | late | early | 0.1 | 8.26659e-05 | [3.76596e-06, 0.000187854] | -0.00392348 | 0.541107 | 0.000522123 | 0.750 | 8.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | late | random | 0.1 | 7.59883e-05 | [7.51307e-06, 0.000176688] | -0.00540243 | 0.475341 | 0.000438491 | 0.875 | 8.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | middle | early | 0.1 | 5.68171e-05 | [-7.49303e-05, 0.000220195] | -0.00272767 | 0.193311 | 8.72871e-05 | 0.667 | 3.0 |
| mistral_7b_v0_1 | wikitext_test | sink_heavy | middle | random | 0.1 | 9.29596e-05 | [1.52002e-05, 0.00017513] | -0.0222656 | 0.189851 | -0.000160597 | 1.000 | 3.0 |

## Value-Content Validation

| model | dataset | control | value intervention | mean KL | mean loss change | mean logit L2 | sites |
|---|---|---|---|---:|---:|---:|---:|
| mistral_7b_v0_1 | wikitext_test | random | ordinary_value_at_sink | 0.183778 | 0.110668 | 42.2662 | 128 |
| mistral_7b_v0_1 | wikitext_test | random | sink_value_at_ordinary | 0.000275958 | -0.00454774 | 4.13669 | 128 |

## Value-Content Paired Asymmetry

Positive paired differences mean ordinary value at the sink was more disruptive than sink value at the ordinary position.

| model | dataset | control | layer band | mean paired KL diff | 95% CI | ordinary-at-sink more disruptive | sequences |
|---|---|---|---|---:|---|---:|---:|
| mistral_7b_v0_1 | wikitext_test | random | all | 0.183502 | [0.021873, 0.398932] | 0.992 | 128.0 |
| mistral_7b_v0_1 | wikitext_test | random | early | 0.312252 | [0.0159051, 0.694022] | 0.986 | 70.0 |
| mistral_7b_v0_1 | wikitext_test | random | late | 0.0309562 | [0.00727761, 0.0710749] | 1.000 | 45.0 |
| mistral_7b_v0_1 | wikitext_test | random | middle | 0.0182767 | [0.00924599, 0.0286246] | 1.000 | 13.0 |

## Secondary All-Head Intervention

| model | dataset | target | delta | mean KL | mean loss change | mean logit L2 | applied sites | skipped sites | sequences |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mistral_7b_v0_1 | wikitext_test | early | 0.05 | 0.000785917 | 0.000304054 | 7.52632 | 244736 | 0 | 64 |
| mistral_7b_v0_1 | wikitext_test | random | 0.05 | 0.000788553 | 0.000664327 | 7.4941 | 244736 | 0 | 64 |
| mistral_7b_v0_1 | wikitext_test | sink | 0.05 | 0.000455104 | 0.00065824 | 5.38716 | 177217 | 67519 | 64 |

## Secondary Paired Differences

Positive paired differences mean the all-head ordinary target was more disruptive than the all-head sink target.

| model | dataset | control | delta | mean control-sink KL | 95% CI | sink less disruptive | sequences |
|---|---|---|---:|---:|---|---:|---:|
| mistral_7b_v0_1 | wikitext_test | early | 0.05 | 0.000330814 | [0.000265352, 0.000396147] | 0.984 | 64.0 |
| mistral_7b_v0_1 | wikitext_test | random | 0.05 | 0.000333449 | [0.000296466, 0.00037573] | 1.000 | 64.0 |
