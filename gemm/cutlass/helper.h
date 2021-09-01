#pragma once

#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

void CheckCudaErrorAux(const char* file, unsigned line)

{

    cudaError_t err = cudaGetLastError();

    if (err == cudaSuccess) return;

    std::cerr << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line

              << std::endl;

    throw std::runtime_error("CUDA ERROR!!!\n");

}



#define CUDA_CHECK_ERROR() CheckCudaErrorAux(__FILE__, __LINE__)