cmake_minimum_required(VERSION 3.17)

project(simd_ut_fma3_avx CXX)

aux_source_directory(. SRC)

add_compile_options(-mavx -mfma)

add_executable(${PROJECT_NAME} ${SRC} ../main.cc)
target_link_libraries(${PROJECT_NAME} "-lgtest")
