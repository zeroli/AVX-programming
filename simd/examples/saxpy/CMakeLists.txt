cmake_minimum_required(VERSION 3.17)

project(saxpy CXX)

aux_source_directory(. SRC)
add_compile_options(-mavx -mavx2 -mfma)
add_executable(${PROJECT_NAME} ${SRC})
