cmake_minimum_required(VERSION 3.17)

project(hello_avx2 CXX)

aux_source_directory(. SRC)
add_compile_options(-mavx2)
add_executable(${PROJECT_NAME} ${SRC})
