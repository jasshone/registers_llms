# Toy-to-Pretrained Relocation Bridge

Question: in toy checkpoints, do source/sink-selective MLP neurons appear and become sufficient to move attention to a dummy/register position, as in the pretrained Pythia relocation experiments?

Setup: select top-k MLP neurons by BOS-position activation excess on train windows. On held-out windows, insert a zero dummy immediately after BOS, patch the selected MLP activations into that dummy using the same max-over-source-positions relocation rule used for pretrained models, and compare against layer-matched random neurons.

Run dir: `outputs/realtext_toy_sink_training/wikitext_pythia_tokenizer_10k_eval128_seed2`
Eval split/examples: `test` / `512`
Top-k / random controls: `32` / `5`
Dummy tokens / scale / fraction: `1` / `10.0` / `1.0`

First checkpoint with positive selected dummy-minus-BOS relocation: step `2000` (0.01319).
Strongest selected relocation: step `7000` with dummy-minus-BOS `0.02175` versus random `-0.03905` and dummy-only `-0.06969`.
Final checkpoint: clean max-head BOS attention `0.2616`, selected score mass `35.87`, selected dummy-minus-BOS `0.007912`, random dummy-minus-BOS `-0.04902`, selected PPL ratio `1.032`.

| step | clean max BOS | score mass | zero BOS ret | random zero ret | dummy-only d-BOS | selected d-BOS | random d-BOS | selected PPL | random PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.03535 | 11.13 | 1.009 | 1.001 | 6.759e-05 | -0.0001391 | 0.0002655 | 1.001 | 0.9985 |
| 1000 | 0.08031 | 19.11 | 0.9308 | 0.9955 | -0.02136 | -0.002787 | -0.009882 | 1.009 | 1.013 |
| 2000 | 0.08984 | 20.96 | 0.9608 | 1.002 | -0.03144 | 0.01319 | -0.01167 | 1.016 | 1.02 |
| 3000 | 0.1095 | 23.06 | 0.9282 | 0.9909 | -0.0442 | -0.0007275 | -0.02267 | 1.008 | 1.06 |
| 4000 | 0.1536 | 25.79 | 0.8677 | 0.9954 | -0.04869 | -0.001051 | -0.03323 | 1.01 | 1.025 |
| 5000 | 0.215 | 28.35 | 0.802 | 1 | -0.05979 | -0.005908 | -0.03991 | 1.011 | 1.035 |
| 6000 | 0.2613 | 30.73 | 0.708 | 0.9984 | -0.06295 | 0.005164 | -0.04038 | 1.014 | 1.055 |
| 7000 | 0.2695 | 32.73 | 0.6612 | 1.004 | -0.06969 | 0.02175 | -0.03905 | 1.024 | 1.034 |
| 8000 | 0.279 | 33.22 | 0.6825 | 1.002 | -0.06755 | -0.007176 | -0.04244 | 1.024 | 1.04 |
| 9000 | 0.2887 | 35.06 | 0.6544 | 0.9999 | -0.06998 | -0.0006325 | -0.05061 | 1 | 1.049 |
| 10000 | 0.2616 | 35.87 | 0.7483 | 1.003 | -0.06703 | 0.007912 | -0.04902 | 1.032 | 1.058 |

Paper-safe interpretation: this bridges the toy and pretrained results if selected toy MLP features become more source-selective over training, ablation reduces BOS attention more than random controls, and the same selected features are sufficient to make a dummy/register slot compete for sink attention. It does not prove the toy and Pythia circuits are identical; it shows the same observable causal motif can arise in the controlled toy setting.
