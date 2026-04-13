# BOS Sink-Neuron Experiments

This codebase turns the notebook prototype into a scripts-first pipeline for sink-neuron experiments on `meta-llama/Meta-Llama-3-8B`.

Typical command order:

```bash
python3 build_windows.py --window-length 1024 --stride 512
python3 measure_bos_sink.py --windows outputs/windows/wikitext103_raw_windows.pt
python3 score_bos_neurons.py --windows outputs/windows/wikitext103_raw_windows.pt
python3 select_bos_neurons.py --scores outputs/scores/bos_scores.pt --percentile 99.98
python3 run_relocation.py --windows outputs/windows/wikitext103_raw_windows.pt --selection outputs/selections/bos_selection.pt
python3 sweep_percentiles.py --windows outputs/windows/wikitext103_raw_windows.pt --scores outputs/scores/bos_scores.pt --percentiles 99.95 99.97 99.98 99.99
```

For long runs, use `tmux` and keep progress bars enabled, e.g. `tmux new -s sink` before launching the commands above.

Key implementation contracts:

- Windows are fixed-length teacher-forced token windows with explicit BOS prepending.
- BOS neuron scores are the mean BOS `silu(gate_proj(x))` activations over all cached windows, reduced to one global `[layer, neuron]` tensor.
- Relocation inserts a zero-initialized dummy embedding slot after BOS and reroutes selected MLP neurons into that slot.
- Baseline and intervened relocation runs always use the same cached window artifact.
