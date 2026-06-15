# BOS Sink-Neuron Experiments

Clean, scripts-first reproduction pipeline for finding sink-neuron candidates, testing relocation/PPL effects, and running rescue sweeps when the first validation pass fails.

## Setup

Use Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA-specific Torch wheels, install the appropriate `torch` build first, then run `pip install -r requirements.txt`. Set `HF_TOKEN` for gated Hugging Face models.

## Main Command

The public entrypoint is:

```bash
python3 run_sink_neuron_pipeline.py
```

The recommended workflow is a per-family sweep. A family sweep includes every configured model in that family, even if previous results had bad PPL, weak relocation, rescue failures, or other undesirable behavior. Failures are written as result rows instead of silently dropping models.

```bash
python3 run_sink_neuron_pipeline.py --list-families

python3 run_sink_neuron_pipeline.py \
  --families gpt2 \
  --run-profile standard \
  --scale-preset core \
  --rescue failed \
  --out outputs/repro_gpt2_family
```

Useful variants:

```bash
# Cheap sanity check over a whole family
python3 run_sink_neuron_pipeline.py --families gpt2 --run-profile smoke --scale-preset core --out /tmp/sink_gpt2_smoke --force

# Multiple full families
python3 run_sink_neuron_pipeline.py --families gpt2 pythia --run-profile standard --scale-preset core --rescue failed

# Exact checkpoint list
python3 run_sink_neuron_pipeline.py --models gpt2_medium pythia_70m --run-profile standard --scale-preset core --rescue failed

# Inspect what would run without loading models
python3 run_sink_neuron_pipeline.py --families pythia --run-profile standard --scale-preset core --dry-run
```

## Families

Print the current list with:

```bash
python3 run_sink_neuron_pipeline.py --list-families
```

Configured families are:

- `gpt2`: `gpt2`, `gpt2_medium`, `gpt2_large`, `gpt2_xl`
- `pythia`: `pythia_70m`, `pythia_160m`, `pythia_410m`, `pythia_1b`, `pythia_1_4b`, `pythia_2_8b`, `pythia_6_9b`, `pythia_12b`
- `llama`: `llama3_2_1b`, `llama3_2_3b`, `llama3_8b`
- `qwen3`: `qwen3_0_6b`, `qwen3_1_7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_30b_a3b`
- `opt`: `opt_125m`, `opt_350m`, `opt_1_3b`, `opt_2_7b`, `opt_6_7b`, `opt_13b`
- `mistral`: `mistral_7b_v0_1`, `mistral_7b_v0_3`, `mistral_nemo_12b`
- `phi`: `phi_1_5`, `phi_2`, `phi3_mini_4k`, `phi3_medium_4k`, `phi3_5_mini`, `phi4`

## Run Profiles And Scales

Run profiles:

- `smoke`: 4 train / 4 validation / 4 test windows, for quick verification
- `standard`: 64 train / 64 validation / 64 test windows, for the main reproduction path

Scale presets:

- `core`: `0.9 1 1.25 1.5 2 3`
- `rescue`: `0.9 1 1.25 1.5 2 3 3.25 3.5 3.75 4 4.5 5 7 8 10`
- `extended`: `0 0.25 0.5 0.6 0.7 0.75 0.8 0.9 1 1.25 1.5 2 2.5 3 3.25 3.5 3.75 4 4.5 5 5.5 6 7 8 10 12 15 16 20 24 30 32`

List scale presets with:

```bash
python3 run_sink_neuron_pipeline.py --list-scale-presets
```

## Optional Model Sets

`--model-set` is a convenience shortcut for curated suites. It is not the main taxonomy; use `--families` when you want every model in a family.

```bash
python3 run_sink_neuron_pipeline.py --list-model-sets
python3 run_sink_neuron_pipeline.py --model-set paper-core --run-profile standard --scale-preset core --rescue failed
python3 run_sink_neuron_pipeline.py --model-set gpt2-pythia --run-profile smoke --scale-preset core
```

Legacy aliases `--preset` and `--model-group` are still accepted for old commands, but new shared commands should prefer `--families`, `--models`, or `--model-set`. Deprecated wrapper entrypoints live in `legacy/`.

## Outputs

Outputs are written under `--out`.

Per model:

- `validation_grid.csv`
- `test_result.csv`
- `summary.md`
- rescue files when rescue is applied

At the output root:

- `final_results.csv`
- `final_results.md`
- per-family CSV/Markdown tables

## Verified Smoke Checks

With the checked-in `requirements.txt`:

- `--model-set gpt2 --run-profile smoke --scale-preset core` produced an `ok` GPT-2 row with `topk=16`, `scale=0.9`, relocation `0.3302`, and PPL ratio `0.99995`.
- `--model-set gpt2-pythia --run-profile smoke --scale-preset core` produced `ok` rows for GPT-2 and Pythia-70M.
- `--model-set gpt2 --run-profile smoke --scale-preset extended` wrote 340 validation rows covering all 32 `extended` scales, both masks, and all standard top-k values.

## Method Contract

- Windows are fixed-length teacher-forced token windows with an explicit BOS/start token.
- Sink discovery includes BOS position 0.
- Candidate scoring uses BOS MLP activations over cached train windows.
- Validation chooses the mask, top-k, and scale; held-out test reports the selected row once.
- Rescue sweeps broaden candidate layer groups and scale/top-k grids for failed rows.
