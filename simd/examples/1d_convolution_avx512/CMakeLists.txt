cmake_minimum_required(VERSION 3.17)

project(1d_convolution_avx512 CXX)

aux_source_directory(. SRC)
add_compile_options(-mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl)
add_executable(${PROJECT_NAME} ${SRC})
