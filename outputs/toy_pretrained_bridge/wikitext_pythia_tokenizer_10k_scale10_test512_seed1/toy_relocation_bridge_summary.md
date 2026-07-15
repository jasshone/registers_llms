# Toy-to-Pretrained Relocation Bridge

Question: in toy checkpoints, do source/sink-selective MLP neurons appear and become sufficient to move attention to a dummy/register position, as in the pretrained Pythia relocation experiments?

Setup: select top-k MLP neurons by BOS-position activation excess on train windows. On held-out windows, insert a zero dummy immediately after BOS, patch the selected MLP activations into that dummy using the same max-over-source-positions relocation rule used for pretrained models, and compare against layer-matched random neurons.

Run dir: `outputs/realtext_toy_sink_training/wikitext_pythia_tokenizer_10k_eval128_seed1`
Eval split/examples: `test` / `512`
Top-k / random controls: `32` / `5`
Dummy tokens / scale / fraction: `1` / `10.0` / `1.0`

First checkpoint with positive selected dummy-minus-BOS relocation: step `0` (5.486e-06).
Strongest selected relocation: step `10000` with dummy-minus-BOS `0.003831` versus random `-0.05532` and dummy-only `-0.07772`.
Final checkpoint: clean max-head BOS attention `0.2535`, selected score mass `36.85`, selected dummy-minus-BOS `0.003831`, random dummy-minus-BOS `-0.05532`, selected PPL ratio `1.015`.

| step | clean max BOS | score mass | zero BOS ret | random zero ret | dummy-only d-BOS | selected d-BOS | random d-BOS | selected PPL | random PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.03567 | 10.75 | 0.9969 | 1 | 0.0001129 | 5.486e-06 | -6.059e-06 | 0.9992 | 1.001 |
| 1000 | 0.07578 | 20.53 | 1 | 1 | -0.01405 | -0.008029 | -0.009314 | 1.005 | 1.01 |
| 2000 | 0.08147 | 25.17 | 0.9018 | 1.014 | -0.02993 | -0.003181 | -0.02016 | 1.014 | 1.012 |
| 3000 | 0.1065 | 26.74 | 0.9986 | 1.002 | -0.04446 | -0.01251 | -0.02354 | 1.013 | 1.016 |
| 4000 | 0.1349 | 28.28 | 0.8929 | 1.001 | -0.05151 | -0.004545 | -0.02825 | 1.017 | 1.022 |
| 5000 | 0.172 | 29.67 | 0.8383 | 1.004 | -0.05856 | -0.00806 | -0.03538 | 1.017 | 1.034 |
| 6000 | 0.2276 | 31.59 | 0.6949 | 1.009 | -0.06481 | -0.003582 | -0.03225 | 1.002 | 1.031 |
| 7000 | 0.2281 | 33.14 | 0.7168 | 1.008 | -0.07129 | -0.002681 | -0.05007 | 1.017 | 1.02 |
| 8000 | 0.2694 | 34.31 | 0.6666 | 1.001 | -0.07428 | 0.001112 | -0.05131 | 1.012 | 1.034 |
| 9000 | 0.2845 | 35.18 | 0.6619 | 1.008 | -0.08171 | -0.0003468 | -0.04763 | 1.01 | 1.041 |
| 10000 | 0.2535 | 36.85 | 0.6538 | 1.005 | -0.07772 | 0.003831 | -0.05532 | 1.015 | 1.043 |

Paper-safe interpretation: this bridges the toy and pretrained results if selected toy MLP features become more source-selective over training, ablation reduces BOS attention more than random controls, and the same selected features are sufficient to make a dummy/register slot compete for sink attention. It does not prove the toy and Pythia circuits are identical; it shows the same observable causal motif can arise in the controlled toy setting.
