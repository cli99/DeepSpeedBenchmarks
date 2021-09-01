#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

#include <benchmark/benchmark.h>

static void CUTLASS_GEMM(benchmark::State& state) {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm75                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //
  int M = state.range(0);
  int N = state.range(1);
  int K = state.range(2);

  float alpha = 1;
  float beta = 0;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::half_t const *ptrA = A.device_data();
  cutlass::half_t const *ptrB = B.device_data();
  cutlass::half_t const *ptrC = C.device_data();
  cutlass::half_t       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  //
  // Launch GEMM on the device
  //

float runtime_ms = 0;
cudaEvent_t startEvent, endEvent;
cudaEventCreate(&startEvent);
cudaEventCreate(&endEvent);

  for (auto _ : state) {
      cudaEventRecord(startEvent);

  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });
  cudaEventRecord(endEvent);
  cudaEventSynchronize(endEvent);


  if (status != cutlass::Status::kSuccess) {
    state.SkipWithError("Error!");
    break ;
  }
  cudaEventElapsedTime(&runtime_ms, startEvent, endEvent);
  state.SetIterationTime(runtime_ms / 10.0e-3);
  }
  const auto flops = 2.0 * M * N * K;
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())*flops);

  return ;
}
BENCHMARK(CUTLASS_GEMM)->UseManualTime()->Unit(benchmark::kMillisecond)->Args({1,5120,20480})->Args({1024,1024,1024});

BENCHMARK_MAIN();