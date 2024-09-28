cmake_minimum_required(VERSION 3.17)

project(permuate CXX)

aux_source_directory(. SRC)

include_directories(../include)
add_compile_options(-mavx -mavx2)
add_executable(${PROJECT_NAME} ${SRC})
