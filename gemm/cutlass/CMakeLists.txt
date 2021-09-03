cmake_minimum_required(VERSION 3.0)
project(DeepSpeedBenchmarks CXX CUDA)

Set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})


if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES benchmark/1.5.3
                      GENERATORS cmake_find_package)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
                    BUILD missing
                    REMOTE conan-center
                    SETTINGS ${settings})

find_package(CUDA)
find_package(benchmark REQUIRED)

cuda_select_nvcc_arch_flags(CUDA_ARCH_OPTS Auto)
list(APPEND NVCC_FLAGS  "${CUDA_ARCH_OPTS}" )

add_executable(gemm_bias_gelu gemm_bias_gelu.cu)
target_include_directories(gemm_bias_gelu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/tools/util/include)
target_link_libraries(gemm_bias_gelu PRIVATE benchmark::benchmark)
target_compile_options(
    gemm_bias_gelu
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)

add_executable(volta_tensorop_gemm volta_tensorop_gemm.cu)
target_include_directories(volta_tensorop_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/tools/util/include)
target_link_libraries(volta_tensorop_gemm PRIVATE benchmark::benchmark)
target_compile_options(
    volta_tensorop_gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)

add_executable(gemm gemm.cu)
target_include_directories(gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/tools/util/include)
target_link_libraries(gemm PRIVATE benchmark::benchmark)
target_compile_options(
    gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)

add_executable(splitk_gemm splitk_gemm.cu)
target_include_directories(splitk_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/tools/util/include)
target_link_libraries(splitk_gemm PRIVATE benchmark::benchmark)
target_compile_options(
    splitk_gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)