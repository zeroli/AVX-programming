cmake_minimum_required(VERSION 3.17)

project(hello_sse CXX)

aux_source_directory(. SRC)
add_compile_options(-msse4.2)
add_executable(${PROJECT_NAME} ${SRC})
