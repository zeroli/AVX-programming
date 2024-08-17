cmake_minimum_required(VERSION 3.17)

project(AVX-Programming)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma")

add_subdirectory(Ch02_01)
add_subdirectory(Ch02_02)
add_subdirectory(permute)
add_subdirectory(shuffle)
add_subdirectory(complex_mul)
