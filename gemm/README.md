# Perfermance comparison


cutlass supports https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md#device-level-gemm


(M = 1, N = 20480, K = 5120) where CUTLASS tensorop has to use (M = 8, N = 20480, K = 5120)

A100
| gemm_bias_relu   | CUTLASS tensorop | DS                           |     |     |
| ---------------- | ---------------- | ---------------------------- | --- | --- |
| fp16, fp16, fp16 | 0.279016         | 0.219034 (0.399258 if M = 8) |     |     |
| fp16, int8, fp16 | X                | 0.110899                     |     |     |

| gemm             | CUTLASS tensorop | DS       |     |     |
| ---------------- | ---------------- | -------- | --- | --- |
| fp16, fp16, fp16 | 0.292403         | 0.219034 |     |     |
| fp16, int8, fp16 | X                | 0.110899 |     |     |


V100
| gemm_bias_relu   | CUTLASS tensorop | DS       |     |     |
| ---------------- | ---------------- | -------- | --- | --- |
| fp16, fp16, fp16 | 0.288256         | 0.307577 |     |     |
| fp16, int8, fp16 | X                | 0.1399   |     |     |


| gemm             | CUTLASS tensorop | DS       |     |     |
| ---------------- | ---------------- | -------- | --- | --- |
| fp16, fp16, fp16 | 0.286652         | 0.304305 |     |     |
| fp16, int8, fp16 | X                | 0.139723 |     |     |