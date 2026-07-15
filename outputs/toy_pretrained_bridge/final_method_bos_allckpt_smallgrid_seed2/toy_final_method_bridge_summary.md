# Toy Final-Method Bridge

This reruns the toy checkpoint bridge with the final paper-figure transfer protocol: emergence-band candidate layers, source-control MLP scoring with source_abs_bonus=0, validation selection over mask/top-k/scale, and fixed source-bias transfer to a zero dummy slot.

| step | mask | k | scale | val reloc | val PPL | test reloc | test PPL | BOS before | dummy after | reason |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | bos_only | 1 | 0.9 | 6.22e-05 | 0.9978 | 8.314e-05 | 0.9996 | 0.03479 | 0.031 | positive_reloc_min_val_ppl |
| 1000 | bos_only | 4 | 16 | 0.0005175 | 1.004 | 0.0001741 | 1.007 | 0.04269 | 0.03093 | positive_reloc_min_val_ppl |
| 2000 | bos_only | 8 | 10 | 0.004323 | 1.006 | 0.00409 | 1.008 | 0.04851 | 0.03513 | positive_reloc_min_val_ppl |
| 3000 | bos_only | 4 | 32 | -0.02044 | 0.9985 | -0.02083 | 1.003 | 0.05853 | 0.02199 | no_positive_max_val_reloc |
| 4000 | bos_only | 1 | 32 | -0.01375 | 1.008 | -0.01285 | 1.007 | 0.06263 | 0.02771 | no_positive_max_val_reloc |
| 5000 | bos_only | 1 | 32 | -0.009807 | 1.006 | -0.008565 | 1.002 | 0.07275 | 0.03616 | no_positive_max_val_reloc |
| 6000 | bos_only | 1 | 32 | -0.007264 | 1.013 | -0.005433 | 1.007 | 0.07741 | 0.03982 | no_positive_max_val_reloc |
| 7000 | bos_only | 1 | 32 | -0.006818 | 1.006 | -0.005005 | 1.003 | 0.08151 | 0.04347 | no_positive_max_val_reloc |
| 8000 | bos_only | 1 | 32 | -0.01032 | 1.026 | -0.0083 | 1.01 | 0.08258 | 0.04185 | no_positive_max_val_reloc |
| 9000 | bos_only | 1 | 32 | -0.008527 | 1.026 | -0.006507 | 1.018 | 0.08334 | 0.04461 | no_positive_max_val_reloc |
| 10000 | bos_only | 1 | 32 | -0.01231 | 1.034 | -0.0103 | 1.017 | 0.08151 | 0.04276 | no_positive_max_val_reloc |

Strongest test relocation: step `2000` with `0.00409` dummy-minus-BOS, PPL ratio `1.008`, mask `bos_only`, k `8`, scale `10`.
Final checkpoint: dummy-minus-BOS `-0.0103`, PPL ratio `1.017`, BOS-before mean `0.08151`, dummy-after mean `0.04276`.
Positive test relocation appears in `3/11` checkpoints under validation-selected rows.
