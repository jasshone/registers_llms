# Toy Final-Method Bridge

This reruns the toy checkpoint bridge with the final paper-figure transfer protocol: emergence-band candidate layers, source-control MLP scoring with source_abs_bonus=0, validation selection over mask/top-k/scale, and fixed source-bias transfer to a zero dummy slot.

| step | mask | k | scale | val reloc | val PPL | test reloc | test PPL | BOS before | dummy after | reason |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | bos_only | 2 | 32 | 0.0002347 | 1 | 0.0002121 | 1.002 | 0.03486 | 0.03116 | positive_reloc_min_val_ppl |
| 1000 | bos_only | 8 | 32 | -0.009754 | 1.006 | -0.01036 | 1.007 | 0.03982 | 0.02454 | no_positive_max_val_reloc |
| 2000 | bos_only | 8 | 32 | -0.003302 | 1.009 | -0.002952 | 1.009 | 0.04903 | 0.03481 | no_positive_max_val_reloc |
| 3000 | bos_only | 4 | 32 | -0.01928 | 1 | -0.01893 | 1.001 | 0.0582 | 0.02583 | no_positive_max_val_reloc |
| 4000 | bos_only | 4 | 32 | -0.01228 | 0.9951 | -0.01158 | 0.9983 | 0.06392 | 0.03141 | no_positive_max_val_reloc |
| 5000 | bos_only | 4 | 32 | -0.01106 | 0.9949 | -0.0107 | 0.9993 | 0.06955 | 0.03424 | no_positive_max_val_reloc |
| 6000 | bos_only | 4 | 32 | -0.00689 | 0.9939 | -0.006388 | 0.9985 | 0.0753 | 0.03899 | no_positive_max_val_reloc |
| 7000 | bos_only | 4 | 32 | -0.007983 | 0.9966 | -0.007158 | 0.9975 | 0.0818 | 0.04183 | no_positive_max_val_reloc |
| 8000 | bos_only | 4 | 32 | -0.005567 | 0.9999 | -0.005012 | 1.005 | 0.08509 | 0.04535 | no_positive_max_val_reloc |
| 9000 | bos_only | 4 | 32 | -0.005332 | 0.9987 | -0.004766 | 1.006 | 0.09168 | 0.04899 | no_positive_max_val_reloc |
| 10000 | bos_only | 1 | 32 | -0.009264 | 1.015 | -0.008402 | 1.015 | 0.08764 | 0.04563 | no_positive_max_val_reloc |

Strongest test relocation: step `0` with `0.0002121` dummy-minus-BOS, PPL ratio `1.002`, mask `bos_only`, k `2`, scale `32`.
Final checkpoint: dummy-minus-BOS `-0.008402`, PPL ratio `1.015`, BOS-before mean `0.08764`, dummy-after mean `0.04563`.
Positive test relocation appears in `1/11` checkpoints under validation-selected rows.
