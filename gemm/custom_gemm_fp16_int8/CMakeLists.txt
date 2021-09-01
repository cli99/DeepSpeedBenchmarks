cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_gemm CXX CUDA)

Set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED HINTS /opt/conda/lib/python3.6/site-packages)

add_executable(custom_gemm custom_gemm.cu)
target_link_libraries(custom_gemm "${TORCH_LIBRARIES}" ${Python_LIBRARIES})

list(APPEND NVCC_FLAGS -arch=sm_80)
target_include_directories(custom_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${Python_INCLUDE_DIRS})
# target_link_libraries(gemm PRIVATE benchmark::benchmark)
target_compile_options(
    custom_gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)

