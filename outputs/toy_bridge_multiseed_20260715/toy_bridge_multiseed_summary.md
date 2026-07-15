# Toy Bridge Multi-Seed Summary

Figure-7-style bridge repeated over seeds 0, 1, and 2. Settings match `outputs/toy_pretrained_bridge/wikitext_pythia_tokenizer_10k_scale10_test512`: Wikitext train text, Pythia-70M tokenizer, test split, 512 selection/eval examples, top-32 selected neurons, five layer-matched random controls, one zero dummy, relocation scale 10.

| step | n | selected d-BOS mean ± sd | random d-BOS mean ± sd | dummy-only d-BOS mean | selected-random mean ± sd | selected > 0 | selected > random | clean max BOS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | -0.000104586 ± 9.75129e-05 | 9.97817e-05 ± 0.000145368 | -6.26532e-06 | -0.000204367 ± 0.000208519 | 1/3 | 1/3 | 0.0355929 |
| 1000 | 3 | 0.00688298 ± 0.0214493 | -0.00114328 ± 0.0146464 | -0.0193478 | 0.00802627 ± 0.00725221 | 1/3 | 3/3 | 0.0754406 |
| 2000 | 3 | 0.01767 ± 0.0234125 | -0.00274835 ± 0.0231911 | -0.0302098 | 0.0204184 ± 0.00403484 | 2/3 | 3/3 | 0.0852003 |
| 3000 | 3 | 0.0132687 ± 0.0349423 | -0.0190182 ± 0.00709321 | -0.0423324 | 0.0322868 ± 0.0279012 | 1/3 | 3/3 | 0.108016 |
| 4000 | 3 | 0.0132695 ± 0.0278847 | -0.0252602 ± 0.00981544 | -0.0487847 | 0.0385297 ± 0.0188183 | 1/3 | 3/3 | 0.13917 |
| 5000 | 3 | 0.00325353 ± 0.0177648 | -0.0279181 ± 0.0169933 | -0.0570111 | 0.0311717 ± 0.00345889 | 1/3 | 3/3 | 0.180688 |
| 6000 | 3 | 0.00848552 ± 0.014026 | -0.0320466 ± 0.00844094 | -0.0610111 | 0.0405321 ± 0.0103133 | 2/3 | 3/3 | 0.220027 |
| 7000 | 3 | 0.0117069 ± 0.0127828 | -0.0369684 ± 0.0142609 | -0.0666164 | 0.0486753 ± 0.0115391 | 2/3 | 3/3 | 0.226018 |
| 8000 | 3 | 0.00676046 ± 0.0174602 | -0.0440643 ± 0.00658354 | -0.0680183 | 0.0508247 ± 0.0148284 | 2/3 | 3/3 | 0.246836 |
| 9000 | 3 | 0.00490316 ± 0.00934163 | -0.0482759 ± 0.0020871 | -0.07238 | 0.053179 ± 0.00799 | 1/3 | 3/3 | 0.261149 |
| 10000 | 3 | 0.00886902 ± 0.00557912 | -0.0460605 ± 0.0110396 | -0.0699677 | 0.0549295 ± 0.00550118 | 3/3 | 3/3 | 0.243277 |

## Notes

- Strongest mean selected relocation: step `2000` with mean selected dummy-minus-BOS `0.01767` and selected-minus-random `0.0204184`.
- Final step `10000`: selected positive in `3/3` seeds and selected beats random in `3/3` seeds.
- This supports a reproducible selected-vs-random bridge effect more strongly than a uniformly positive absolute relocation effect: seeds 1 and 2 are weaker than seed 0, but selected exceeds layer-matched random controls at late checkpoints.
