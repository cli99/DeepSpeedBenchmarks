cmake_minimum_required(VERSION 2.8)
project(DeepSpeedBenchmarks CXX CUDA)

Set(CMAKE_CXX_STANDARD 17)

add_executable(gemm cutlass/gemm.cu)

list(APPEND NVCC_FLAGS -arch=sm_80)
target_include_directories(gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include)
target_compile_options(
    gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)