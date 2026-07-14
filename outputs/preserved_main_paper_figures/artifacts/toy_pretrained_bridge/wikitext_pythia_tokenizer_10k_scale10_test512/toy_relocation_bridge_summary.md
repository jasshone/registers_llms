# Toy-to-Pretrained Relocation Bridge

Question: in toy checkpoints, do source/sink-selective MLP neurons appear and become sufficient to move attention to a dummy/register position, as in the pretrained Pythia relocation experiments?

Setup: select top-k MLP neurons by BOS-position activation excess on train windows. On held-out windows, insert a zero dummy immediately after BOS, patch the selected MLP activations into that dummy using the same max-over-source-positions relocation rule used for pretrained models, and compare against layer-matched random neurons.

Run dir: `outputs/realtext_toy_sink_training/wikitext_pythia_tokenizer_10k_eval128`
Eval split/examples: `test` / `512`
Top-k / random controls: `32` / `5`
Dummy tokens / scale / fraction: `1` / `10.0` / `1.0`

First checkpoint with positive selected dummy-minus-BOS relocation: step `1000` (0.03146).
Strongest selected relocation: step `3000` with dummy-minus-BOS `0.05304` versus random `-0.01084` and dummy-only `-0.03834`.
Final checkpoint: clean max-head BOS attention `0.2148`, selected score mass `24.51`, selected dummy-minus-BOS `0.01486`, random dummy-minus-BOS `-0.03384`, selected PPL ratio `1.059`.

| step | clean max BOS | score mass | zero BOS ret | random zero ret | dummy-only d-BOS | selected d-BOS | random d-BOS | selected PPL | random PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.03576 | 11.4 | 1 | 1 | -0.0001993 | -0.0001802 | 3.987e-05 | 0.9953 | 0.9988 |
| 1000 | 0.07023 | 17.92 | 1 | 1 | -0.02263 | 0.03146 | 0.01577 | 1.021 | 1.025 |
| 2000 | 0.0843 | 17.75 | 0.7903 | 1.01 | -0.02927 | 0.043 | 0.02358 | 1.065 | 1.072 |
| 3000 | 0.1081 | 19.98 | 0.7522 | 0.9637 | -0.03834 | 0.05304 | -0.01084 | 1.1 | 1.06 |
| 4000 | 0.129 | 22.13 | 0.7185 | 0.9851 | -0.04616 | 0.0454 | -0.0143 | 1.093 | 1.062 |
| 5000 | 0.155 | 23.27 | 0.7403 | 0.9976 | -0.05269 | 0.02373 | -0.008471 | 1.048 | 1.108 |
| 6000 | 0.1712 | 23.21 | 0.7014 | 0.9975 | -0.05528 | 0.02387 | -0.0235 | 1.061 | 1.09 |
| 7000 | 0.1805 | 23.9 | 0.7791 | 0.9926 | -0.05887 | 0.01605 | -0.02178 | 1.055 | 1.156 |
| 8000 | 0.1921 | 24.29 | 0.7239 | 0.9968 | -0.06222 | 0.02635 | -0.03845 | 1.055 | 1.052 |
| 9000 | 0.2102 | 23.85 | 0.7382 | 0.9955 | -0.06545 | 0.01569 | -0.04658 | 1.053 | 1.055 |
| 10000 | 0.2148 | 24.51 | 0.7165 | 1.012 | -0.06515 | 0.01486 | -0.03384 | 1.059 | 1.09 |

Paper-safe interpretation: this bridges the toy and pretrained results if selected toy MLP features become more source-selective over training, ablation reduces BOS attention more than random controls, and the same selected features are sufficient to make a dummy/register slot compete for sink attention. It does not prove the toy and Pythia circuits are identical; it shows the same observable causal motif can arise in the controlled toy setting.
