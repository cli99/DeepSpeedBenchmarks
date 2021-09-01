cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

Set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED HINTS /opt/conda/lib/python3.6/site-packages)

add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
