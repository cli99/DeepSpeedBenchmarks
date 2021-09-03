# Perfermance comparison


cutlass supports https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md#device-level-gemm


(M = 1, N = 20480, K = 5120) where CUTLASS tensorop has to use (M = 8, N = 20480, K = 5120 or M = 20480, N = 8, K = 5120)

## A100
| gemm_bias_relu   | CUTLASS tensorop | DS                           |     |     |
| ---------------- | ---------------- | ---------------------------- | --- | --- |
| fp16, fp16, fp16 | 0.163373        | 0.219034 (0.399258 if M = 8) |     |     |
| fp16, int8, fp16 | X                | 0.110899                     |     |     |

| gemm             | CUTLASS tensorop | DS       |     |     |
| ---------------- | ---------------- | -------- | --- | --- |
| fp16, fp16, fp16 | 0.16189        | 0.215782 (0.396147 if M = 8) |     |     |
| fp16, int8, fp16 | X                | 0.108721 |     |     |


best cutlass tile config:
num_stages = 6

```cpp
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<64, 64, 16>;  // <- threadblock tile M = 128, N =
                                             // 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<32, 32, 16>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 4
```


## V100
| gemm_bias_relu   | CUTLASS tensorop | DS       |     |     |
| ---------------- | ---------------- | -------- | --- | --- |
| fp16, fp16, fp16 | 0.237783         | 0.307577 (0.655784 if M = 8) |     |     |
| fp16, int8, fp16 | X                | 0.1399   |     |     |


| gemm             | CUTLASS tensorop | DS       |     |     |
| ---------------- | ---------------- | -------- | --- | --- |
| fp16, fp16, fp16 | 0.236701         | 0.304305 (0.636116 if M = 8) |     |     |
| fp16, int8, fp16 | X                | 0.139723 |     |     |


best cutlass tile config:
num_stages = 2

```cpp
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N =
                                             // 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 32,
                             32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4
```