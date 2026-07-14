# Toy Training BOS Position Sweep

Each model is trained from scratch with the BOS token placed at a fixed sequence position. Metrics compare learned attention to absolute position 0 against attention to the trained BOS position.

| BOS train pos | final loss | final PPL | attn to pos0 | attn to BOS pos | attn to next pos | BOS-pos0 | BOS-next |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.1320 | 5.05e+05 | 0.0610 | 0.0610 | 0.0104 | +0.0000 | +0.0505 |
| 2 | 13.2021 | 5.41e+05 | 0.0177 | 0.0191 | 0.0117 | +0.0014 | +0.0074 |
| 4 | 13.1249 | 5.01e+05 | 0.0203 | 0.0115 | 0.0134 | -0.0089 | -0.0019 |
| 8 | 13.2226 | 5.53e+05 | 0.0195 | 0.0059 | 0.0115 | -0.0137 | -0.0056 |
| 16 | 13.0192 | 4.51e+05 | 0.0196 | 0.0038 | 0.0111 | -0.0158 | -0.0073 |
| 32 | 13.0238 | 4.53e+05 | 0.0186 | 0.0017 | 0.0105 | -0.0170 | -0.0089 |
| 64 | 12.9879 | 4.37e+05 | 0.0178 | 0.0015 | 0.0141 | -0.0164 | -0.0127 |
