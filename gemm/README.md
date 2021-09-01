# Perfermance comparison

(M = 1, N = 20480, K = 5120) where CUTLASS has to use (M = 8, N = 20480, K = 5120)

|     gemm_bias_relu    |     CUTLASS(ms)    |     DS(ms)      |   |   |
|-----------------------|--------------------|-----------------|---|---|
|     fp16 fp16         |     0.292403       |     0.219034    |   |   |
|     fp16, int8        |     X              |     0.110899    |   |   |