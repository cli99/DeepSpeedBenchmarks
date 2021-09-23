cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_gemm CXX CUDA)

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

find_package(benchmark REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED HINTS /opt/conda/lib/python3.8/site-packages)

add_executable(custom_gemm custom_gemm.cu)
target_link_libraries(custom_gemm PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES} benchmark::benchmark)

cuda_select_nvcc_arch_flags(CUDA_ARCH_OPTS Auto)


list(APPEND NVCC_FLAGS  "${CUDA_ARCH_OPTS}" )
target_include_directories(custom_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${Python_INCLUDE_DIRS})
# target_link_libraries(gemm PRIVATE benchmark::benchmark)
target_compile_options(
    custom_gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
)

