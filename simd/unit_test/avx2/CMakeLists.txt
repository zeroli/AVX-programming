cmake_minimum_required(VERSION 3.17)

project(simd_ut_avx2 CXX)

aux_source_directory(. SRC)

add_compile_options(-mavx -mavx2)

add_executable(${PROJECT_NAME} ${SRC} ../main.cc)
target_link_libraries(${PROJECT_NAME} "-lgtest")
