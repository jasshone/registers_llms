# Final Split Bias Transfer Tables

## GPT2 Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| gpt2 | bos_only | 4 | 3 | 1:1 2:3 | 0.3006 | 1.0024 | +0.24% |

## PYTHIA Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| pythia_410m | bos_only | 4 | 3 | 5:4 | 0.3159 | 1.0178 | +1.78% |

## LLAMA Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| llama3_2_1b | bos_only | 4 | 1 | 0:2 1:2 | 0.3966 | 1.0901 | +9.01% |

## Qwen3 Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| qwen3_0_6b | SKIPPED |  |  |  |  |  | ValueError('no positive validation relocation; best relocation=-0.000163391, ppl |
| qwen3_1_7b | bos_only | 1 | 3 | 2:1 | 0.4477 | 0.9768 | -2.32% |

## OPT Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| opt_125m | bos_only | 8 | 3 | 0:8 | 0.4428 | 1.0092 | +0.92% |

## MISTRAL Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| mistral_7b_v0_1 | bos_only | 1 | 2 | 1:1 | 0.3383 | 0.9943 | -0.57% |

## PHI Family

| model | mask | k | scale | layers | reloc | ppl ratio | ppl change |
|---|---|---:|---:|---|---:|---:|---:|
| phi_1_5 | bos_only | 2 | 3 | 0:1 1:1 | 0.4921 | 1.0023 | +0.23% |

