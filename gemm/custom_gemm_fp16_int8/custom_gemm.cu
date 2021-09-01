#include <limits>
// #include "custom_cuda_layers.h"

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

// https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L44
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
  const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
  const float* weight_cast = reinterpret_cast<const float*>(weight);

  output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
  weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
  vals_cast += (unsigned)(blockIdx.x / outputBlocks) * (hidden_dim >> 1);

  // reading all the quantization scale into a small shared buffer
  __shared__ float shared_quantize_scale[MAX_QUANTIZE_GROUPING];
  if (threadIdx.x < ((groups << merge_count)))
    shared_quantize_scale[threadIdx.x] = (qscale[threadIdx.x]);
  __syncthreads();
  unsigned hidden_half = (hidden_dim >> 1);
  unsigned merge_hidden = hidden_dim >> merge_count;
  unsigned total_hidden = (merge_hidden << block_reduce);
  for (int j = 0; j < input_size; j++) {
    __half2 sum[2];
#pragma unroll
    for (int t = 0; t < 2; t++) {
      sum[t] = __float2half2_rn(0.f);
    }

    {
      int wid = gid << loop_unroll_bits;
      weight_cast +=
          wid * output_size + (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
      unsigned merge_hidden_total = 0;
      while (wid < hidden_dim) {
        unsigned w_index = wid / merge_hidden;

        __half2 vals_f[INPUT_TILE1 * loop_unroll];
        {
          __half2 val_h[loop_unroll >> 1];
          val_h[0] =
              vals_cast[(j) * (hidden_half << block_reduce) + (wid >> 1)];
          val_h[1] =
              vals_cast[(j) * (hidden_half << block_reduce) + (wid >> 1) + 1];

          __half* inp_data[2];
          inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
          inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);
          vals_f[0] = __halves2half2(inp_data[0][0], inp_data[0][0]);
          vals_f[1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
          vals_f[2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
          vals_f[3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
        }

        int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;

        if (col < output_size) {
          __half2 weight_h[loop_unroll << 1];
          {
            float weight_q[loop_unroll];
#pragma unroll
            for (int k = 0; k < loop_unroll; k++)
              weight_q[k] = weight_cast[k * output_size];

            merge_hidden_total +=
                (wid / (merge_hidden_total + merge_hidden)) * merge_hidden;
            unsigned q_index =
                (wid - merge_hidden_total) + (col << 2) * total_hidden +
                (unsigned)(blockIdx.x / outputBlocks) * merge_hidden;

            float scale_data[4];
#pragma unroll
            for (int k = 0; k < loop_unroll; k++)
              scale_data[k] =
                  shared_quantize_scale[(((q_index + k * total_hidden) /
                                          quantization_stride)
                                         << merge_count) +
                                        w_index];

            for (int k = 0; k < loop_unroll; k++) {
              int8_t* weight_8 = reinterpret_cast<int8_t*>(&weight_q[k]);
              __half* weight_data =
                  reinterpret_cast<__half*>(&weight_h[k << 1]);
              weight_data[0] = __float2half((float)weight_8[0] * scale_data[0]);
              weight_data[1] = __float2half((float)weight_8[1] * scale_data[1]);
              weight_data[2] = __float2half((float)weight_8[2] * scale_data[2]);
              weight_data[3] = __float2half((float)weight_8[3] * scale_data[3]);
            }
          }
#pragma unroll
          for (int li = 0; li < loop_unroll; li++) {
            sum[0] += vals_f[li] * weight_h[(li << 1)];
            sum[1] += vals_f[li] * weight_h[(li << 1) + 1];
          }
        }
        wid += warp_num << loop_unroll_bits;
        weight_cast += (output_size * (warp_num << loop_unroll_bits));
      }
    }
    {
      const float2* bias_cast;
      if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
      __shared__ __half2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 2)];

      {
        partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1)] = sum[0];
        partial_result[(gid << 1) * (WARP_SIZE + 2) + (lane << 1) + 1] = sum[1];
        __syncthreads();

        sum[0] = (partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1)]);
        sum[1] =
            (partial_result[(lane << 1) * (WARP_SIZE + 2) + (gid << 1) + 1]);

        float2 sum_f[2];
        sum_f[0] = __half22float2(sum[0]);
        sum_f[1] = __half22float2(sum[1]);
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
          sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
          sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
          sum_f[1].x += g.shfl_xor(sum_f[1].x, i);
          sum_f[1].y += g.shfl_xor(sum_f[1].y, i);
        }

        if (lane == 0) {
          partial_result[(gid << 1)] = __float22half2_rn(sum_f[0]);
          partial_result[(gid << 1) + 1] = __float22half2_rn(sum_f[1]);
        }
        __syncthreads();

        if (gid == 0) {
          sum[0] = partial_result[(lane << 1)];
          sum[1] = partial_result[(lane << 1) + 1];
        }
      }

      if (gid == 0 && (gid + j) < input_size) {
        int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
        if (col < output_size) {
          if (bias && blockIdx.x < outputBlocks) {
            float2 bias_ff = bias_cast[col];
            __half2* bias_h = reinterpret_cast<__half2*>(&bias_ff);
            float2 bias_f[2];
            bias_f[0] = __half22float2(bias_h[0]);
            bias_f[1] = __half22float2(bias_h[1]);
            float2 sum_f[2];
            sum_f[0] = __half22float2(sum[0]);
            sum_f[1] = __half22float2(sum[1]);
            sum_f[0].x += bias_f[0].x;
            sum_f[0].y += bias_f[0].y;
            sum_f[1].x += bias_f[1].x;
            sum_f[1].y += bias_f[1].y;
            if (add_gelu && gridDim.x == outputBlocks) {
              sum_f[0].x = gelu(sum_f[0].x);
              sum_f[0].y = gelu(sum_f[0].y);
              sum_f[1].x = gelu(sum_f[1].x);
              sum_f[1].y = gelu(sum_f[1].y);
            }
            sum[0] = __float22half2_rn(sum_f[0]);
            sum[1] = __float22half2_rn(sum_f[1]);
          }
          float2 result;
          __half2* result_h = reinterpret_cast<__half2*>(&result);
          result_h[0] = sum[0];
          result_h[1] = sum[1];
          output_cast[col + (j) * (output_size << block_reduce)] = result;
        }
      }
    }
    weight_cast = reinterpret_cast<const float*>(weight);
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

// https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/csrc/transformer/inference_specialized/csrc/custom_gemm.cu#L819
template <typename T>
void launch_input_tiled_gemm_kernel_v2(benchmark::State& state, T* output,
                                       const T* vals, const int8_t* weight,
                                       const T* bias, unsigned int hidden_dim,
                                       unsigned int input_size,
                                       unsigned int output_size, float* scale,
                                       unsigned int groups,
                                       unsigned int merge_count, T* block_sums,
                                       bool add_gelu, cudaStream_t stream) {
  output_size /= 4;
  int outputBlocks = (output_size - 1) / WARP_SIZE + 1;

  int block_reduce = (SMs > outputBlocks ? SMs / outputBlocks : 1);
  int br2 = (int)log2(block_reduce);

  block_reduce = (int)pow(2.0, (float)br2);

  hidden_dim /= block_reduce;

  constexpr int threads = 1024;
  int blockStride = output_size * hidden_dim;

  dim3 grid_dim(outputBlocks * block_reduce);
  dim3 block_dim(threads);

  cudaEvent_t startEvent, endEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);

  for (auto _ : state) {
    cudaEventRecord(startEvent, stream);

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

    CUDA_CHECK(cudaEventRecord(endEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(endEvent));

    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, startEvent, endEvent);
    state.SetIterationTime(runtime_ms / 10.0e3);
  }
}

void run(benchmark::State& state) {
  // https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/deepspeed/ops/transformer/inference/transformer_inference.py#L289

  auto hidden_size = 5120;
  torch::Tensor input =
      torch::rand({1, 1, hidden_size}, torch::TensorOptions()
                                           .dtype(torch::kFloat16)
                                           .layout(torch::kStrided)
                                           .device(torch::kCUDA));
  torch::Tensor weight =
      torch::rand({hidden_size, 4 * hidden_size}, torch::TensorOptions()
                                                      .dtype(torch::kFloat16)
                                                      .layout(torch::kStrided)
                                                      .device(torch::kCUDA)
                                                      .requires_grad(true));

  torch::Tensor q_scale = torch::rand({2}, torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .layout(torch::kStrided)
                                               .device(torch::kCUDA));  // TODO:

  int groups = 1;       // TODO:
  int merge_count = 1;  // TODO:

  auto input_cont = input.contiguous();
  auto options = torch::TensorOptions()
                     .dtype(input_cont.options().dtype())
                     .layout(torch::kStrided)
                     .device(torch::kCUDA)
                     .requires_grad(false);

  auto output = torch::empty(
      {input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
  int bsz = input_cont.size(0) * input_cont.size(1);

  // computing the blocking across K dimension
  int out_blocks = (weight.size(1) - 1) / CACHLINE + 1;
  out_blocks = (out_blocks < SMs) ? (SMs / out_blocks) : 1;
  int br2 = (int)log2(out_blocks);
  out_blocks = (int)pow(2.0, (float)br2);

  auto block_sums = torch::empty(
      {input_cont.size(0) * out_blocks, input_cont.size(1), weight.size(1)},
      options);

  using T = __half;
  // https://github.com/microsoft/DeepSpeed-internal/blob/inference-specialized-only/csrc/transformer/inference_specialized/csrc/pt_binding.cpp#L646
  launch_input_tiled_gemm_kernel_v2(
      state, (T*)output.data_ptr(), (T*)input_cont.data_ptr(),
      (int8_t*)weight.data_ptr(), (T*)nullptr, input_cont.size(2), bsz,
      weight.size(1), (float*)q_scale.data_ptr(), groups, merge_count,
      (T*)block_sums.data_ptr(), false, Context::Instance().GetCurrentStream());
}

BENCHMARK(run)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();