#include <limits>
// #include "custom_cuda_layers.h"

#include <ATen/cuda/CUDABlas.h>
#include <benchmark/benchmark.h>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <torch/extension.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "context.h"

namespace cg = cooperative_groups;

#define INPUT_TILE 1
#define INPUT_TILE1 1

// Input tile used in the gemm kernel v2
#define INPUT_TILE2_Q 8

#define INPUT_TILE2 8

#define MAX_REG_SIZE 20

// https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/csrc/transformer/inference_specialized/includes/custom_cuda_layers.h#L11
#define WARP_SIZE 32
#define SMs 160
#define CACHLINE 128
#define MAX_REGISTERS 256

#define MAX_WARP_NUM 32
#define MAX_BLOCK_SUM 8

#define loop_unroll 4
#define loop_unroll_bits 2

#define inner_loop_unroll 4
#define inner_loop_unroll_bits 2

#define INT8WIDTH 2

#define MAX_QUANTIZE_GROUPING 1024

#define ACC_HALF true

inline __device__ float gelu(const float x) {
  float y = 0.5 * x *
            (1.0 + tanhf(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
  return y;
}

void CheckCudaErrorAux(const char* file, unsigned line)

{
  cudaError_t err = cudaGetLastError();

  if (err == cudaSuccess) return;

  std::cerr << cudaGetErrorString(err) << "(" << err << ") at " << file << ":"
            << line

            << std::endl;

  throw std::runtime_error("CUDA ERROR!!!\n");
}

#define CUDA_CHECK_ERROR() CheckCudaErrorAux(__FILE__, __LINE__)

// https://github.com/microsoft/DeepSpeed-internal/blob/reyazda/fast-attn/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L43
__global__ void input_tiled_gemm_kernel_v2(
    __half* output, const __half* vals, const int8_t* weight,
    const __half* bias, unsigned hidden_dim, unsigned block_reduce,
    unsigned input_size, unsigned output_size, unsigned outputBlocks,
    unsigned blockStride, float* qscale, unsigned groups, __half* block_sums,
    unsigned merge_count = 1, unsigned quantization_stride = 1,
    bool add_gelu = false) {
  // #if __CUDA_ARCH__ >= 700
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  unsigned int gid = threadIdx.x >> 5;
  unsigned int lane = threadIdx.x & 0x1f;

  int warp_num = blockDim.x >> 5;

  float2* output_cast = reinterpret_cast<float2*>(
      ((gridDim.x == outputBlocks) ? output : block_sums));
  const float4* vals_cast = reinterpret_cast<const float4*>(vals);
  const float2* qscale_cast = reinterpret_cast<const float2*>(qscale);
  const float4* weight_cast = reinterpret_cast<const float4*>(weight);

  output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
  weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
  vals_cast += (unsigned)(blockIdx.x / outputBlocks) * (hidden_dim >> 3);
  int output_size_quarter = output_size >> 2;
  // reading all the quantization scale into a small shared buffer
  __shared__ float2 shared_quantize_scale[MAX_QUANTIZE_GROUPING];

  __shared__ float2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 2)];
  if ((threadIdx.x << 1) < ((groups << merge_count)))
    shared_quantize_scale[threadIdx.x] = (qscale_cast[threadIdx.x]);
  __syncthreads();
  unsigned hidden_quarter = (hidden_dim >> 2);
  // for (int j = 0; j < input_size; j++)
  {
    float2 sum[2];
#pragma unroll
    for (int t = 0; t < 2; t++) {
      sum[t].x = 0.f;
      sum[t].y = 0.f;
    }

    {
      weight_cast += (gid << 3) * output_size_quarter +
                     (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
      int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
      float4 weight_q[2];
      if (col < output_size) {
        weight_q[0] = weight_cast[0];
        weight_q[1] = weight_cast[output_size];
      }
      float4 val_h;
      val_h = vals_cast[gid];
      weight_cast += (output_size_quarter * (warp_num << 3));
      int iterations = hidden_dim / (WARP_SIZE << 3) - 1;
      for (int u = 0; u < iterations; u++) {
        if (col < output_size) {
          float4 w_q[2];
#pragma unroll
          for (int m = 0; m < 2; m++) {
            w_q[m] = weight_q[m];
            weight_q[m] = weight_cast[m * output_size];
          }

          __half* inp_data = (__half*)(&val_h);
          int8_t* weight_8 = reinterpret_cast<int8_t*>(w_q);
#pragma unroll
          for (int li = 0; li < 8; li++) {
            float inp_f = inp_data[li];
            sum[0].x += inp_f * weight_8[0];
            sum[0].y += inp_f * weight_8[1];
            sum[1].x += inp_f * weight_8[2];
            sum[1].y += inp_f * weight_8[3];
            weight_8 += 4;
          }
        }
        val_h = vals_cast[gid + (u << 5)];
        weight_cast += (output_size_quarter * (warp_num << 3));
      }
      __half* inp_data = (__half*)(&val_h);
      int8_t* weight_8 = reinterpret_cast<int8_t*>(weight_q);
#pragma unroll
      for (int li = 0; li < 8; li++) {
        float inp_f = inp_data[li];
        sum[0].x += inp_f * weight_8[0];
        sum[0].y += inp_f * weight_8[1];
        sum[1].x += inp_f * weight_8[2];
        sum[1].y += inp_f * weight_8[3];
        weight_8 += 4;
      }
      // quantization scaling
      {
        unsigned q_index = (gid << 2) + (col << 2) * hidden_dim;
        unsigned new_index = q_index / quantization_stride;
        float2 t_scale = shared_quantize_scale[new_index];
        float* scale_f = (float*)&t_scale;
        sum[0].x *= scale_f[0];
        sum[0].y *=
            scale_f[((q_index + hidden_dim) / quantization_stride) - new_index];
        sum[1].x *= scale_f[((q_index + hidden_dim * 2) / quantization_stride) -
                            new_index];
        sum[1].y *= scale_f[((q_index + hidden_dim * 3) / quantization_stride) -
                            new_index];
      }
    }
    {
      const float2* bias_cast;
      if (bias) bias_cast = reinterpret_cast<const float2*>(bias);

      {
        partial_result[gid * (WARP_SIZE + 1) + lane] = sum[0];
        partial_result[(gid + warp_num) * (WARP_SIZE + 1) + lane] = sum[1];
        __syncthreads();

        sum[0] = partial_result[lane * (WARP_SIZE + 2) + gid];
        sum[1] = partial_result[(lane + warp_num) * (WARP_SIZE + 1) + gid];

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
          sum[0].x += g.shfl_xor(sum[0].x, i);
          sum[0].y += g.shfl_xor(sum[0].y, i);
          sum[1].x += g.shfl_xor(sum[1].x, i);
          sum[1].y += g.shfl_xor(sum[1].y, i);
        }

        if (lane == 0) {
          partial_result[gid] = sum[0];
          partial_result[gid + WARP_SIZE] = sum[1];
        }
        __syncthreads();

        if (gid == 0) {
          sum[0] = partial_result[lane];
          sum[1] = partial_result[lane + WARP_SIZE];
        }
      }

      if (gid == 0) {
        int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
        if (col < output_size) {
          if (bias && blockIdx.x < outputBlocks) {
            float2 bias_ff = bias_cast[col];
            __half2* bias_h = reinterpret_cast<__half2*>(&bias_ff);
            float2 bias_f[2];
            bias_f[0] = __half22float2(bias_h[0]);
            bias_f[1] = __half22float2(bias_h[1]);
            sum[0].x += bias_f[0].x;
            sum[0].y += bias_f[0].y;
            sum[1].x += bias_f[1].x;
            sum[1].y += bias_f[1].y;
            if (add_gelu && gridDim.x == outputBlocks) {
              sum[0].x = gelu(sum[0].x);
              sum[0].y = gelu(sum[0].y);
              sum[1].x = gelu(sum[1].x);
              sum[1].y = gelu(sum[1].y);
            }
          }
          float2 result;
          __half2* result_h = reinterpret_cast<__half2*>(&result);
          result_h[0] = __float22half2_rn(sum[0]);
          result_h[1] = __float22half2_rn(sum[1]);
          output_cast[col] = result;
        }
      }
    }
  }
  // #endif
}

// https://github.com/microsoft/DeepSpeed-internal/blob/reyazda/fast-attn/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L714
__global__ void input_tiled_gemm_kernel_v2(
    __half* output, const __half* vals, const __half* weight,
    const __half* bias, __half* block_sums, unsigned int hidden_dim,
    unsigned int block_reduce, unsigned int input_size,
    unsigned int output_size, unsigned int outputBlocks,
    unsigned int blockStride, bool add_gelu = false) {
  // #if __CUDA_ARCH__ >= 700
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  unsigned int gid = threadIdx.x >> 5;
  unsigned int lane = threadIdx.x & 0x1f;

  int warp_num = blockDim.x >> 5;

  __half2* output_cast = reinterpret_cast<__half2*>(
      ((gridDim.x == outputBlocks) ? output : block_sums));
  const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
  const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);
  output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
  int hidden_half = hidden_dim >> 1;
  weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
  vals_cast += (unsigned)(blockIdx.x / outputBlocks) * hidden_half;

  for (int j = 0; j < input_size; j += (INPUT_TILE2)) {
    __half2 sum[INPUT_TILE2];
#pragma unroll
    for (int t = 0; t < INPUT_TILE2; t++) {
      sum[t] = __float2half2_rn(0.f);
    }

    {
      int wid = gid << loop_unroll_bits;
      weight_cast +=
          wid * output_size + (blockIdx.x % outputBlocks) * WARP_SIZE + lane;

      while (wid < hidden_dim) {
        __shared__ __half2
            vals_h[(loop_unroll >> 1) * INPUT_TILE2 * MAX_WARP_NUM];
        {
          // we read (loop_unroll >> 2) half-2 values per lane, and for 2 times
          // of the INPUT_TILE this makes more threads engaged in reading data
          // from shared memory into registers!
          if (lane < (INPUT_TILE2 << 1)) {
            if (((lane >> 1) + j) < input_size) {
              // here, we consider loop_unroll is always higher that 4!
              unsigned int inp_id = ((lane % 2) << (loop_unroll_bits - 2));

              unsigned int offset =
                  (j + (lane >> 1)) * (block_reduce * (hidden_dim >> 1)) +
                  inp_id;
#pragma unroll
              for (int li = 0; li < (loop_unroll >> 2); li++) {
                vals_h[li + inp_id + (((lane >> 1) << (loop_unroll_bits - 1))) +
                       (gid << (loop_unroll_bits - 1)) * INPUT_TILE2] =
                    vals_cast[offset + (wid >> 1) + li];
              }
            }
          }
          g.sync();
        }

        int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;

        if (col < output_size) {
          __half2 weight_h[loop_unroll];
#pragma unroll
          for (int k = 0; k < loop_unroll; k++)
            weight_h[k] = weight_cast[output_size * k];
          auto internal_offset = (gid << (loop_unroll_bits - 1)) * INPUT_TILE2;
#pragma unroll
          for (int t = 0; t < INPUT_TILE2 && (t + j) < input_size; t++) {
            __half2* base_input =
                vals_h + (t << (loop_unroll_bits - 1)) + internal_offset;
#pragma unroll
            for (int li = 0; li < (loop_unroll >> 1); li++) {
              __half* inp_data = reinterpret_cast<__half*>(base_input + li);
              sum[t] += __halves2half2(inp_data[0], inp_data[0]) *
                        weight_h[(li << 1)];
              sum[t] += __halves2half2(inp_data[1], inp_data[1]) *
                        weight_h[(li << 1) + 1];
            }
          }
        }
        wid += warp_num << loop_unroll_bits;
        weight_cast += (output_size * (warp_num << loop_unroll_bits));
      }
    }
    {
      const __half2* bias_cast;
      if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
      __shared__ __half2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 2)];

      for (int t = 0; t < INPUT_TILE2; t += 2) {
        if ((t + j) < input_size) {
          partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1)] = sum[t];
          partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1) + 1] =
              sum[t + 1];
          b.sync();

          float2 sum_f[2];
          sum_f[0] = __half22float2(
              partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1)]);
          sum_f[1] = __half22float2(
              partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1) + 1]);

#pragma unroll
          for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
            sum_f[1].y += g.shfl_xor(sum_f[1].y, i);
            sum_f[1].x += g.shfl_xor(sum_f[1].x, i);
            sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
          }

          if (lane == 0) {
            partial_result[(gid << 1)] = __float22half2_rn(sum_f[0]);
            partial_result[(gid << 1) + 1] = __float22half2_rn(sum_f[1]);
          }
          b.sync();

          if (gid == (t >> 1)) {
            sum[t] = partial_result[(lane << 1)];
            sum[t + 1] = partial_result[(lane << 1) + 1];
          }
        }
      }

      if ((gid << 1) < INPUT_TILE2 && ((gid << 1) + j) < input_size) {
        int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
        if (col < output_size) {
          if (bias && blockIdx.x < outputBlocks) {
            __half2 bias_h = bias_cast[col];
            float2 bias_f = __half22float2(bias_h);
            float2 sum_f[2];
            sum_f[0] = __half22float2(sum[(gid << 1)]);
            sum_f[1] = __half22float2(sum[(gid << 1) + 1]);
            sum_f[0].x += bias_f.x;
            sum_f[0].y += bias_f.y;
            sum_f[1].x += bias_f.x;
            sum_f[1].y += bias_f.y;
            if (add_gelu && gridDim.x == outputBlocks) {
              sum_f[0].x = gelu(sum_f[0].x);
              sum_f[0].y = gelu(sum_f[0].y);
              sum_f[1].x = gelu(sum_f[1].x);
              sum_f[1].y = gelu(sum_f[1].y);
            }
            sum[(gid << 1)] = __float22half2_rn(sum_f[0]);
            sum[(gid << 1) + 1] = __float22half2_rn(sum_f[1]);
          }
          output_cast[col + (j + (gid << 1)) * (block_reduce * output_size)] =
              (sum[(gid << 1)]);
          if (((gid << 1) + j + 1) < input_size)
            output_cast[col +
                        (j + (gid << 1) + 1) * (block_reduce * output_size)] =
                (sum[(gid << 1) + 1]);
        }
      }
    }
    weight_cast = reinterpret_cast<const __half2*>(weight);
    weight_cast += ((blockIdx.x / outputBlocks) * blockStride);
  }
  // #endif
}

// https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L767
__global__ void block_reduce_kernel(__half* output, __half* block_sums,
                                    unsigned batch, unsigned int output_size,
                                    bool add_gelu = false) {
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
  unsigned total_count = batch * output_size;
  unsigned int gid = threadIdx.x >> 5;
  unsigned int lane = threadIdx.x & 0x1f;
  unsigned int warp_num = blockDim.x >> 5;

  __half2* output_cast = reinterpret_cast<__half2*>(output);
  __half2* block_sums_cast = reinterpret_cast<__half2*>(block_sums);

  unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
  block_sums_cast += gid * output_size;

  if (col_index < total_count) {
    __shared__ __half2 data_shared[MAX_WARP_NUM * (WARP_SIZE + 1)];

    data_shared[gid * (WARP_SIZE) + lane] =
        block_sums_cast[(col_index / output_size) * (warp_num * output_size) +
                        col_index % output_size];

    b.sync();

    float2 data = __half22float2(
        data_shared[(lane % warp_num) * WARP_SIZE +
                    gid * (WARP_SIZE / warp_num) + (lane / warp_num)]);

    b.sync();
#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
      data.x += g.shfl_xor(data.x, i);
      data.y += g.shfl_xor(data.y, i);
    }

    if ((lane % warp_num) == 0) {
      if (add_gelu) {
        data.x = gelu(data.x);
        data.y = gelu(data.y);
      }
      data_shared[gid * (WARP_SIZE / warp_num) + (lane / warp_num)] =
          __float22half2_rn(data);
    }

    b.sync();

    if (gid == 0) output_cast[col_index] = data_shared[lane];
  }
}

// https://github.com/microsoft/DeepSpeed-internal/blob/reyazda/fast-attn/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L982
template <typename T>
void launch_input_tiled_gemm_kernel_v2(
    T* output, const T* vals, const int8_t* weight, const T* bias,
    unsigned int hidden_dim, unsigned int input_size, unsigned int output_size,
    float* scale, unsigned int groups, unsigned int merge_count, T* block_sums,
    bool add_gelu, cudaStream_t stream) {
  output_size /= 4;
  int outputBlocks = (output_size - 1) / WARP_SIZE + 1;

  int block_reduce = 2;  //(SMs > outputBlocks ? SMs / outputBlocks : 1);
  int br2 = 1;           //(int)log2(block_reduce);
                         //
  // block_reduce = (int)pow(2.0, (float)br2);

  hidden_dim /= block_reduce;

  constexpr int threads = 1024;
  int blockStride = (output_size >> 2) * hidden_dim;

  dim3 grid_dim(outputBlocks * block_reduce);
  dim3 block_dim(threads);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  input_tiled_gemm_kernel_v2<<<grid_dim, block_dim, 0, stream>>>(
      output, vals, weight, bias, hidden_dim, br2, input_size, output_size,
      outputBlocks, blockStride, scale, groups, block_sums, merge_count,
      (((hidden_dim << br2) >> (merge_count)) * (output_size << 2)) / groups,
      add_gelu);
  if (block_reduce > 1) {
    output_size <<= 1;
    dim3 grids(((output_size * input_size) - 1) / WARP_SIZE + 1);
    dim3 blocks(block_reduce * WARP_SIZE);
    block_reduce_kernel<<<grids, blocks, 0, stream>>>(
        output, block_sums, input_size, (output_size), add_gelu);
  }
}

// https://github.com/microsoft/DeepSpeed-internal/blob/reyazda/fast-attn/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L1094
template <typename T>
void launch_input_tiled_gemm_kernel_v2(T* output, const T* vals,
                                       const T* weight, const T* bias,
                                       T* block_sums, unsigned int hidden_dim,
                                       unsigned int input_size,
                                       unsigned int output_size, bool add_gelu,
                                       cudaStream_t stream) {
  output_size /= 2;
  int outputBlocks = (output_size - 1) / WARP_SIZE + 1;

  int block_reduce = (SMs > outputBlocks ? SMs / outputBlocks : 1);
  int br2 = (int)log2(block_reduce);
  block_reduce = (int)pow(2.0, (float)br2);

  constexpr int threads = 1024;
  int blockStride = (output_size * hidden_dim) / block_reduce;

  dim3 grid_dim(outputBlocks * block_reduce);
  dim3 block_dim(threads);
  input_tiled_gemm_kernel_v2<<<grid_dim, block_dim, 0, stream>>>(
      output, vals, weight, bias, block_sums, hidden_dim / block_reduce,
      block_reduce, input_size, output_size, outputBlocks, blockStride,
      add_gelu);
  if (block_reduce > 1) {
    dim3 grids(((output_size * input_size) - 1) / WARP_SIZE + 1);
    dim3 blocks(block_reduce * WARP_SIZE);
    block_reduce_kernel<<<grids, blocks, 0, stream>>>(
        output, block_sums, input_size, (output_size), add_gelu);
  }
}

__global__ void input_tiled_gemm_kernel(__half* output, const __half* vals,
                                        const __half* weight,
                                        const __half* bias, int hidden_dim,
                                        int input_size, int output_size) {
#if __CUDA_ARCH__ >= 700

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  int gid = threadIdx.x >> 5;
  int lane = threadIdx.x & 0x1f;
  int warp_num = blockDim.x >> 5;

  __half2* output_cast = reinterpret_cast<__half2*>(output);
  const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
  const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);

  unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
  int hidden_half = hidden_dim >> 1;

  {
    __half2 sum;
    sum = __float2half2_rn(0.f);

    {
      int wid = gid << loop_unroll_bits;
      weight_cast += (wid * output_size + col_index);

      while (wid < hidden_dim) {
        __half2 vals_f[loop_unroll * (INPUT_TILE1)];
        {
          {
            {
              __half2 val_h[2];
              val_h[0] = vals_cast[(wid >> 1)];
              val_h[1] = vals_cast[(wid >> 1) + 1];

              __half* inp_data[2];
              inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
              inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

              vals_f[0] = __halves2half2(inp_data[0][0], inp_data[0][0]);
              vals_f[1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
              vals_f[2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
              vals_f[3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
            }
          }
        }

        if (col_index < output_size) {
          __half2 weight_h[loop_unroll];
#pragma unroll
          for (int k = 0; k < loop_unroll; k++) {
            if ((k + wid) < hidden_dim)
              weight_h[k] = weight_cast[k * output_size];
            else
              weight_h[k] = __float2half2_rn(0.f);
          }

#pragma unroll
          for (int k = 0; k < (loop_unroll >> inner_loop_unroll_bits); k++) {
            {
              {
#pragma unroll
                for (int li = 0; li < inner_loop_unroll; li++) {
                  weight_h[0] = (vals_f[li] * weight_h[li]);
                  if (ACC_HALF)
                    sum += weight_h[0];
                  else {
                    float2 weight_f = __half22float2(weight_h[0]);
                    float2 sum_f = __half22float2(sum);
                    sum_f.x += weight_f.x;
                    sum_f.y += weight_f.y;
                    sum = __float22half2_rn(sum_f);
                  }
                }
              }
            }
          }
        }
        wid += (warp_num << loop_unroll_bits);
        weight_cast += (output_size * (warp_num << loop_unroll_bits));
      }
    }
    {
      const __half2* bias_cast;
      if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
      __shared__ __half2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

      {
        {
          __half2 sum_g = sum;
          partial_result[gid * (WARP_SIZE + 1) + lane] = sum;

          b.sync();
          float2 sum_f;
          sum_f = __half22float2(partial_result[lane * (WARP_SIZE + 1) + gid]);

          b.sync();
#pragma unroll
          for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum_f.x += g.shfl_xor(sum_f.x, i);
            sum_f.y += g.shfl_xor(sum_f.y, i);
          }

          if (lane == 0) {
            partial_result[gid] = __float22half2_rn(sum_f);
          }

          b.sync();

          if (gid == 0) {
            int col = blockIdx.x * WARP_SIZE + lane;
            if (col < output_size) {
              sum_g = partial_result[lane];
              if (bias) {
                float2 bias_f = __half22float2(bias_cast[col]);
                sum_f = __half22float2(sum_g);
                sum_f.x += bias_f.x;
                sum_f.y += bias_f.y;
                sum_g = __float22half2_rn(sum_f);
              }
              output_cast[col + output_size] = (sum_g);
            }
          }
        }
      }
    }
    weight_cast = reinterpret_cast<const __half2*>(weight);
  }
#endif
}

__global__ void input_tiled_gemm_kernel(float* output, const float* vals,
                                        const float* weight, const float* bias,
                                        int hidden_dim, int input_size,
                                        int output_size) {
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  int gid = threadIdx.x >> 5;
  int lane = threadIdx.x & 0x1f;
  int warp_num = blockDim.x >> 5;

  float2* output_cast = reinterpret_cast<float2*>(output);
  const float2* vals_cast = reinterpret_cast<const float2*>(vals);
  const float2* weight_cast = reinterpret_cast<const float2*>(weight);

  int hidden_half = hidden_dim >> 1;

  for (int j = 0; j < input_size; j += (INPUT_TILE1)) {
    float2 sum[INPUT_TILE1];
#pragma unroll
    for (int t = 0; t < (INPUT_TILE1); t++) {
      sum[t].x = 0;
      sum[t].y = 0;
    }

    {
      int wid = gid << 1;
      int offset = wid * output_size;

      while (wid < hidden_dim) {
        float2 val_data[INPUT_TILE1];
        {
          for (int t = 0; t < INPUT_TILE1; t++) {
            if ((t + j) < input_size) {
              val_data[t] = vals_cast[(j + t) * hidden_half + (wid >> 1)];
            }
          }
        }

        int row = blockIdx.x * WARP_SIZE + lane;
        auto offset1 = offset + row;
        while (row < output_size) {
          float2 weight[2];
          weight[0] = weight_cast[offset1];
          weight[1] = weight_cast[output_size + offset1];

          for (int t = 0; t < INPUT_TILE1; t++) {
            if ((t + j) < input_size) {
              float2 mul[2];
              mul[0].x = val_data[t].x * weight[0].x;
              mul[0].y = val_data[t].x * weight[0].y;
              mul[1].x = val_data[t].y * weight[1].x;
              mul[1].y = val_data[t].y * weight[1].y;

              sum[t].x += mul[0].x + mul[1].x;
              sum[t].y += mul[0].y + mul[1].y;
            }
          }
          row += (gridDim.x * WARP_SIZE);
          offset1 += (gridDim.x * WARP_SIZE);
        }
        wid += warp_num * 2;
        offset += (output_size * warp_num * 2);
      }
    }
    {
      const float2* bias_cast;
      if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
      __shared__ float2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

      for (int t = 0; t < (INPUT_TILE1); t++) {
        if ((t + j) < input_size) {
          float2 sum_g = sum[t];
          partial_result[gid * (WARP_SIZE + 1) + lane] = sum_g;
          __syncthreads();

          sum_g = partial_result[lane * (WARP_SIZE + 1) + gid];
          __syncthreads();

#pragma unroll
          for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum_g.x += g.shfl_xor(sum_g.x, i);
            sum_g.y += g.shfl_xor(sum_g.y, i);
          }

          if (lane == 0) {
            partial_result[gid] = sum_g;
          }

          __syncthreads();

          if (gid == 0) {
            int col = blockIdx.x * WARP_SIZE + lane;
            if (col < output_size) {
              sum_g = partial_result[lane];
              if (bias) {
                float2 bias_f = bias_cast[col];
                sum_g.x += bias_f.x;
                sum_g.y += bias_f.y;
              }
              output_cast[col + (j + t) * output_size] = sum_g;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void launch_input_tiled_gemm_kernel(T* output, const T* vals, const T* weight,
                                    const T* bias, int hidden_dim,
                                    int input_size, int output_size,
                                    cudaStream_t stream) {
  constexpr int threads = 1024;
  output_size /= 2;
  dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
  dim3 block_dim(threads);
  input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
      output, vals, weight, bias, hidden_dim, input_size, output_size);
}

template void launch_input_tiled_gemm_kernel(float* output, const float* vals,
                                             const float* weight,
                                             const float* bias, int hidden_dim,
                                             int input_size, int output_size,
                                             cudaStream_t stream);

template void launch_input_tiled_gemm_kernel(__half* output, const __half* vals,
                                             const __half* weight,
                                             const __half* bias, int hidden_dim,
                                             int input_size, int output_size,
                                             cudaStream_t stream);

#define INPUT_TILE11 10

__global__ void input_tiled_gemm_kernel_v2_qqq(
    __half* output, const __half* vals, const int8_t* weight,
    const __half* bias, unsigned hidden_dim, unsigned block_reduce,
    unsigned input_size, unsigned output_size, unsigned outputBlocks,
    unsigned blockStride, float* qscale, unsigned groups, __half* block_sums,
    unsigned merge_count = 1, unsigned quantization_stride = 1,
    bool add_gelu = false) {
#if __CUDA_ARCH__ >= 700
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  unsigned int gid = threadIdx.x >> 5;
  unsigned int lane = threadIdx.x & 0x1f;

  int warp_num = blockDim.x >> 5;

  // reading all the quantization scale into a small shared buffer
  extern __shared__ float base1[];
  float2* shared_quantize_scale = (float2*)base1;
  __half2* shared_sum =
      (__half2*)&shared_quantize_scale[MAX_QUANTIZE_GROUPING >> 1] + 2112 +
      (gid * (WARP_SIZE + 1)) + lane;

  //__shared__ float2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 2)];

  // for (int j = 0; j < input_size; j++)
  {
    const float4* vals_cast = reinterpret_cast<const float4*>(vals);
    const float2* qscale_cast = reinterpret_cast<const float2*>(qscale);
    const float4* weight_cast = reinterpret_cast<const float4*>(weight);

    weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
    vals_cast += (unsigned)(blockIdx.x / outputBlocks) * (hidden_dim >> 3) +
                 lane * (((hidden_dim >> 3) << block_reduce));
    int output_size_quarter = output_size >> 2;

    if ((threadIdx.x << 1) < ((groups << merge_count)))
      shared_quantize_scale[threadIdx.x] = (qscale_cast[threadIdx.x]);
    __syncthreads();
    unsigned hidden_quarter = (hidden_dim >> 2);

    {
      weight_cast += (gid << 3) * output_size_quarter +
                     (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
      int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
      float4 weight_q[2];
      if (col < output_size) {
        weight_q[0] = weight_cast[0];
        weight_q[1] = weight_cast[output_size];
      }
      float4* vals_h_shared =
          (float4*)(&shared_quantize_scale[(MAX_QUANTIZE_GROUPING >> 1)]) +
          (gid << 4);
      {
        // we read (loop_unroll >> 2) half-2 values per lane, and for 2 times of
        // the INPUT_TILE this makes more threads engaged in reading data from
        // shared memory into registers!
        if (lane < (INPUT_TILE11) && (lane) < input_size) {
          vals_h_shared[lane] = vals_cast[gid];
        }
        g.sync();
      }
      weight_cast += (output_size_quarter * (warp_num << 3));
      int iterations = hidden_dim / (warp_num << 3) - 2;
      __half2 summation[INPUT_TILE11];

      float4 w_q[2];
      {
        if (col < output_size) {
          int8_t* weight_8 = (int8_t*)(weight_q);
          for (int t = 0; t < INPUT_TILE11; t++) {
            __half* input_sh = (__half*)(vals_h_shared + t);
            __half2 inp = __halves2half2(input_sh[0], input_sh[0]);
            __half2 weight1 = __halves2half2((__half)((float)weight_8[0]),
                                             (__half)((float)weight_8[1]));
            __half2 weight2 = __halves2half2((__half)((float)weight_8[2]),
                                             (__half)((float)weight_8[3]));
            ((__half2*)w_q)[t] = inp * weight1;
            summation[t] = inp * weight2;
          }
          for (int li = 1; li < 4; li++) {
            __half2 weight1 =
                __halves2half2((__half)((float)weight_8[(li << 2)]),
                               (__half)((float)weight_8[(li << 2) + 1]));
            __half2 weight2 =
                __halves2half2((__half)((float)weight_8[(li << 2) + 2]),
                               (__half)((float)weight_8[(li << 2) + 3]));
            for (int t = 0; t < INPUT_TILE11; t++) {
              __half* input_sh = (__half*)(vals_h_shared + t);
              __half2 inp = __halves2half2(input_sh[li], input_sh[li]);
              ((__half2*)w_q)[t] += inp * weight1;
              summation[t] += inp * weight2;
            }
          }

          weight_q[0] = weight_cast[0];

          for (int li = 4; li < 8; li++) {
            __half2 weight1 =
                __halves2half2((__half)((float)weight_8[(li << 2)]),
                               (__half)((float)weight_8[(li << 2) + 1]));
            __half2 weight2 =
                __halves2half2((__half)((float)weight_8[(li << 2) + 2]),
                               (__half)((float)weight_8[(li << 2) + 3]));
            for (int t = 0; t < INPUT_TILE11; t++) {
              __half* input_sh = (__half*)(vals_h_shared + t);
              __half2 inp = __halves2half2(input_sh[li], input_sh[li]);
              ((__half2*)w_q)[t] += inp * weight1;
              summation[t] += inp * weight2;
            }
          }
          weight_q[1] = weight_cast[output_size];
        }
        vals_cast += warp_num;
        weight_cast += (output_size_quarter * (warp_num << 3));
        if (lane < (INPUT_TILE11) && (lane) < input_size) {
          vals_h_shared[lane] = vals_cast[gid];
        }
        g.sync();
      }
      for (int u = 0; u < iterations; u++) {
        if (col < output_size) {
          int8_t* weight_8 = (int8_t*)(weight_q);
          for (int li = 0; li < 4; li++) {
            __half2 weight1 =
                __halves2half2((__half)((float)weight_8[(li << 2)]),
                               (__half)((float)weight_8[(li << 2) + 1]));
            __half2 weight2 =
                __halves2half2((__half)((float)weight_8[(li << 2) + 2]),
                               (__half)((float)weight_8[(li << 2) + 3]));
            for (int t = 0; t < INPUT_TILE11; t++) {
              __half* input_sh = (__half*)(vals_h_shared + t);
              __half2 inp = __halves2half2(input_sh[li], input_sh[li]);
              ((__half2*)w_q)[t] += inp * weight1;
              summation[t] += inp * weight2;
            }
          }
          weight_q[0] = weight_cast[0];
          for (int li = 4; li < 8; li++) {
            __half2 weight1 =
                __halves2half2((__half)((float)weight_8[(li << 2)]),
                               (__half)((float)weight_8[(li << 2) + 1]));
            __half2 weight2 =
                __halves2half2((__half)((float)weight_8[(li << 2) + 2]),
                               (__half)((float)weight_8[(li << 2) + 3]));
            for (int t = 0; t < INPUT_TILE11; t++) {
              __half* input_sh = (__half*)(vals_h_shared + t);
              __half2 inp = __halves2half2(input_sh[li], input_sh[li]);
              ((__half2*)w_q)[t] += inp * weight1;
              summation[t] += inp * weight2;
            }
          }
          weight_q[1] = weight_cast[output_size];
        }
        vals_cast += warp_num;
        weight_cast += (output_size_quarter * (warp_num << 3));

        vals_h_shared[lane] =
            lane < input_size ? vals_cast[gid] : vals_h_shared[lane];
        g.sync();
      }

      int8_t* weight_8 = (int8_t*)(weight_q);
      for (int li = 0; li < 8; li++) {
        __half2 weight1 =
            __halves2half2((__half)((float)weight_8[(li << 2)]),
                           (__half)((float)weight_8[(li << 2) + 1]));
        __half2 weight2 =
            __halves2half2((__half)((float)weight_8[(li << 2) + 2]),
                           (__half)((float)weight_8[(li << 2) + 3]));
        for (int t = 0; t < INPUT_TILE11; t++) {
          __half* input_sh = (__half*)(vals_h_shared + t);
          __half2 inp = __halves2half2(input_sh[li], input_sh[li]);
          ((__half2*)w_q)[t] += inp * weight1;
          summation[t] += inp * weight2;
        }
        for (int t = 0; t < INPUT_TILE11; t++) {
          shared_sum[(t << 11)] = ((__half2*)w_q)[t];
          shared_sum[(t << 11) + blockDim.x] = summation[t];
        }
      }
    }
    {
      int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
      const float2* bias_cast;
      if (bias) bias_cast = reinterpret_cast<const float2*>(bias);

      float2* output_cast = reinterpret_cast<float2*>(
          ((gridDim.x == outputBlocks) ? output : block_sums));
      output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
      __half2* partial_result =
          (__half2*)&shared_quantize_scale[MAX_QUANTIZE_GROUPING >> 1] + 2112 +
          lane * (WARP_SIZE + 1) + gid;

      __syncthreads();
      // quantization scaling

      unsigned q_index = (gid << 2) + (col << 2) * hidden_dim;
      unsigned new_index = q_index / quantization_stride;
      unsigned index[3];
      index[0] = ((q_index + hidden_dim) / quantization_stride) - new_index;
      index[1] =
          ((q_index + (hidden_dim << 1)) / quantization_stride) - new_index;
      index[2] = ((q_index + hidden_dim * 3) / quantization_stride) - new_index;
      float scale_f[2];
      scale_f[0] = shared_quantize_scale[new_index].x;
      scale_f[1] = shared_quantize_scale[new_index].y;

      shared_sum =
          (__half2*)(shared_quantize_scale + (MAX_QUANTIZE_GROUPING >> 1));
      for (int t = 0; t < INPUT_TILE11; t++) {
        __half2 sum_f[2];
        sum_f[0] = (partial_result[t << 11]);
        sum_f[1] = (partial_result[(t << 11) + blockDim.x]);
        sum_f[0].x *= scale_f[0];
        sum_f[0].y *= (index[0] == 0 ? scale_f[0] : scale_f[1]);
        sum_f[1].x *= (index[1] == 0 ? scale_f[0] : scale_f[1]);
        sum_f[1].y *= (index[2] == 0 ? scale_f[0] : scale_f[1]);
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
          sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
          sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
          sum_f[1].x += g.shfl_xor(sum_f[1].x, i);
          sum_f[1].y += g.shfl_xor(sum_f[1].y, i);
        }

        if (lane == 0) {
          shared_sum[gid] = sum_f[0];
          shared_sum[gid + WARP_SIZE] = sum_f[1];
        }
      }

      __syncthreads();
      if (gid < input_size) {
        if (col < output_size) {
          __half2 sum_f[2];
          sum_f[0] = shared_sum[lane];
          sum_f[1] = shared_sum[lane + WARP_SIZE];
          if (bias && blockIdx.x < outputBlocks) {
            float2 bias_ff = bias_cast[col];
            __half2* bias_h = (__half2*)(&bias_ff);
            sum_f[0].x += bias_h[0].x;
            sum_f[0].y += bias_h[0].y;
            sum_f[1].x += bias_h[1].x;
            sum_f[1].y += bias_h[1].y;
            if (add_gelu && gridDim.x == outputBlocks) {
              sum_f[0].x = gelu(sum_f[0].x);
              sum_f[0].y = gelu(sum_f[0].y);
              sum_f[1].x = gelu(sum_f[1].x);
              sum_f[1].y = gelu(sum_f[1].y);
            }
          }
          float2 result;
          __half2* result_h = reinterpret_cast<__half2*>(&result);
          result_h[0] = (sum_f[0]);
          result_h[1] = (sum_f[1]);
          output_cast[col + (gid * (output_size << block_reduce))] = result;
        }
      }
    }
  }
#endif
}

template <typename T>
void launch_input_tiled_gemm_kernel_v3(
    T* output, const T* vals, const int8_t* weight, const T* bias,
    unsigned int hidden_dim, unsigned int input_size, unsigned int output_size,
    float* scale, unsigned int groups, unsigned int merge_count, T* block_sums,
    bool add_gelu, cudaStream_t stream) {
  output_size /= 4;
  int outputBlocks = (output_size - 1) / WARP_SIZE + 1;
  // printf("outputBlocks =  %d , output_size = %d \n", outputBlocks,
  // output_size);
  int block_reduce = (SMs > outputBlocks ? SMs / outputBlocks : 1);
  int br2 = 1;  //(int)log2(block_reduce);
                //
  block_reduce = (int)pow(2.0, (float)br2);

  int threads = ((hidden_dim / block_reduce) >> 1);
  if (threads > 1024) threads = 1024;
  int blockStride = ((output_size >> 2) * hidden_dim) / block_reduce;

  // printf("block_reduce =  %d , br2 = %d , groups= %d, threads = %d, hidden:
  // %d, output: %d\n", block_reduce, br2,
  //                groups, threads, hidden_dim, output_size);
  dim3 grid_dim(outputBlocks * block_reduce);
  dim3 block_dim(threads);
  cudaFuncSetAttribute(input_tiled_gemm_kernel_v2_qqq,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, 98160);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  input_tiled_gemm_kernel_v2_qqq<<<grid_dim, block_dim, 98160, stream>>>(
      output, vals, weight, bias, hidden_dim / block_reduce, br2, input_size,
      output_size, outputBlocks, blockStride, scale, groups, block_sums,
      merge_count, ((hidden_dim >> merge_count) * (output_size << 2)) / groups,
      add_gelu);
  if (block_reduce > 1) {
    output_size <<= 1;
    dim3 grids(((output_size * input_size) - 1) / WARP_SIZE + 1);
    dim3 blocks(block_reduce * WARP_SIZE);
    block_reduce_kernel<<<grids, blocks, 0, stream>>>(
        output, block_sums, input_size, (output_size), add_gelu);
  }
}

template void launch_input_tiled_gemm_kernel_v3(
    __half* output, const __half* vals, const int8_t* weight,
    const __half* bias, unsigned int hidden_dim, unsigned int input_size,
    unsigned int output_size, float* scale, unsigned int groups,
    unsigned int merge_count, __half* block_sums, bool add_gelu,
    cudaStream_t stream);

/*
launch_input_tiled_gemm_kernel_v3((T*)output.data_ptr(),
                                              (T*)input.data_ptr(),
                                              (int8_t*)weight.data_ptr(),
                                              (T*)nullptr,
                                              input.size(2),
                                              bsz,
                                              weight.size(1),
                                              (float*)q_scale.data_ptr(),
                                              groups,
                                              0,
                                              (T*)(workspace + buff_size),
                                              false,
                                              Context::Instance().GetCurrentStream());
*/
template <typename T>
void allocat_workspace(unsigned hidden_dim, unsigned max_seq_len,
                       unsigned batch_size, unsigned head_size = 128) {
  size_t _workSpaceSize = 3 * (hidden_dim * batch_size * max_seq_len);
  Context::Instance().GenWorkSpace(_workSpaceSize, sizeof(T));
}

int main(benchmark::State& state) {
  // https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/deepspeed/ops/transformer/inference/transformer_inference.py#L289

  auto hidden_size = 5120;
  torch::Tensor input =
      torch::ones({4, 1, hidden_size}, torch::TensorOptions()
                                           .dtype(torch::kFloat16)
                                           .layout(torch::kStrided)
                                           .device(torch::kCUDA));
  torch::Tensor weight =
      torch::ones({hidden_size, 4 * hidden_size}, torch::TensorOptions()
                                                      .dtype(torch::kInt8)
                                                      .layout(torch::kStrided)
                                                      .device(torch::kCUDA)
                                                      .requires_grad(false));
  // cublas gemm output
  auto output_cublas =
      at::matmul(input, weight.to(torch::kFloat16)).to(torch::kFloat32);

  torch::Tensor q_scale = torch::ones({1}, torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .layout(torch::kStrided)
                                               .device(torch::kCUDA));

  int groups = 1;
  int merge_count = 0;

  auto input_cont = input.contiguous();
  auto options = torch::TensorOptions()
                     .dtype(input_cont.options().dtype())
                     .layout(torch::kStrided)
                     .device(torch::kCUDA)
                     .requires_grad(false);

  int bsz = input.size(0) * input.size(1);

  using T = __half;

  auto workspace = Context::Instance().GetWorkSpace();
  if (!workspace) {
    allocat_workspace<T>(input.size(2), 256, input.size(0));
    workspace = Context::Instance().GetWorkSpace();
  }

  size_t buff_size = Context::Instance().get_workspace_size() / 3;

  // computing the blocking across K dimension
  int out_blocks = (weight.size(1) - 1) / CACHLINE + 1;
  out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
  int br2 = (int)log2(out_blocks);
  out_blocks = (int)pow(2.0, (float)br2);

  auto block_sums = torch::zeros(
      {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)},
      options);

  torch::Tensor bias =
      torch::ones({1, hidden_size}, torch::TensorOptions()
                                        .dtype(torch::kFloat16)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA));

  int cnt = 1;
  float total = 0;
  for (int i = 0; i < cnt; i++) {
    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    // auto output = torch::from_blob(
    //     workspace, {input.size(0), input.size(1), weight.size(1)}, options);

    auto output =
        torch::zeros({input.size(0), input.size(1), weight.size(1)}, options);
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    // https://github.com/microsoft/DeepSpeed-internal/blob/reyazda/fast-attn/csrc/transformer/inference_specialized/csrc/pt_binding.cpp#L688
    launch_input_tiled_gemm_kernel_v3(
        (T*)output.data_ptr(), (T*)input.data_ptr(), (int8_t*)weight.data_ptr(),
        (T*)nullptr, input.size(2), bsz, weight.size(1),
        (float*)q_scale.data_ptr(), groups, merge_count,
        (T*)(workspace + buff_size), false,
        Context::Instance().GetCurrentStream());
    CUDA_CHECK_ERROR();
    CUDA_CHECK(cudaEventRecord(endEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(endEvent));

    output = output.to(torch::kFloat32);
    std::cout << "at 0,0,0 = " << output.index({0, 0, 0}) << "\n";
    std::cout << "at 0,0,1 = " << output.index({0, 0, 1}) << "\n";
    std::cout << "at 0,0,2 = " << output.index({0, 0, 2}) << "\n";
    std::cout << "at 1,0,0 = " << output.index({1, 0, 0}) << "\n";
    std::cout << "at 1,0,1 = " << output.index({1, 0, 1}) << "\n";
    std::cout << "at 1,0,2 = " << output.index({1, 0, 2}) << "\n";

    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, startEvent, endEvent);
    // state.SetIterationTime(runtime_ms / 10.0e3);
    std::cout << "runtime_ms a = " << runtime_ms << " ms\n";
    if (i != 1) {
      total += runtime_ms;
    }
  }
  std::cout << "average runtime_ms = " << total / (cnt - 1) << " ms\n";

  return 0;
}

// void run_fp16(benchmark::State& state) {
//   auto hidden_size = 5120;
//   torch::Tensor input =
//       torch::rand({1, 1, hidden_size}, torch::TensorOptions()
//                                            .dtype(torch::kFloat16)
//                                            .layout(torch::kStrided)
//                                            .device(torch::kCUDA));
//   torch::Tensor weight =
//       torch::rand({hidden_size, 4 * hidden_size}, torch::TensorOptions()
//                                                       .dtype(torch::kFloat16)
//                                                       .layout(torch::kStrided)
//                                                       .device(torch::kCUDA)
//                                                       .requires_grad(true));
//   auto options = torch::TensorOptions()
//                      .dtype(input.options().dtype())
//                      .layout(torch::kStrided)
//                      .device(torch::kCUDA)
//                      .requires_grad(false);

//   using T = __half;

//   auto workspace = Context::Instance().GetWorkSpace();
//   if (!workspace) {
//     allocat_workspace<T>(input.size(2), 256, input.size(0));
//     workspace = Context::Instance().GetWorkSpace();
//   }
//   auto output = torch::from_blob(
//       workspace, {input.size(0), input.size(1), weight.size(1)}, options);

//   size_t buff_size = Context::Instance().get_workspace_size() / 3;

//   int bsz = input.size(0) * input.size(1);

//   int out_blocks = (weight.size(1) - 1) / (CACHLINE >> 1) + 1;
//   out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
//   int br2 = (int)log2(out_blocks);
//   out_blocks = (int)pow(2.0, (float)br2);

//   auto block_sums = torch::from_blob(
//       (workspace + buff_size),
//       {input.size(0) * out_blocks, input.size(1), weight.size(1)}, options);

//   torch::Tensor bias =
//       torch::rand({1, 4 * hidden_size}, torch::TensorOptions()
//                                             .dtype(torch::kFloat16)
//                                             .layout(torch::kStrided)
//                                             .device(torch::kCUDA));

//   cudaEvent_t startEvent, endEvent;
//   cudaEventCreate(&startEvent);
//   cudaEventCreate(&endEvent);

//   int cnt = 100;
//   float total = 0;
//   for (int i = 0; i < cnt; i++) {
//     CUDA_CHECK(cudaEventRecord(startEvent, 0));

//     //
//     https://github.com/microsoft/DeepSpeed-internal/blob/reyazda/fast-attn/csrc/transformer/inference_specialized/csrc/pt_binding.cpp#L549

//     launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
//     (T*)input.data_ptr(),
//                                    (T*)weight.data_ptr(),
//                                    (T*)bias.data_ptr(), input.size(2), bsz,
//                                    weight.size(1),
//                                    Context::Instance().GetCurrentStream());
//     CUDA_CHECK_ERROR();
//     CUDA_CHECK(cudaEventRecord(endEvent, 0));
//     CUDA_CHECK(cudaEventSynchronize(endEvent));

//     float runtime_ms = 0;
//     cudaEventElapsedTime(&runtime_ms, startEvent, endEvent);
//     // state.SetIterationTime(runtime_ms / 10.0e3);
//     // std::cout << "runtime_ms = " << runtime_ms << " ms\n";
//     if (i != 1) {
//       total += runtime_ms;
//     }
//   }
//   std::cout << "average runtime_ms = " << total / (cnt - 1) << " ms\n";
// }

// BENCHMARK(run_int8)->UseManualTime()->Unit(benchmark::kMillisecond);

// BENCHMARK_MAIN();