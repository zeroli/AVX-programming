cmake_minimum_required(VERSION 3.17)

project(simd_ut_avx CXX)

aux_source_directory(. SRC)

add_compile_definitions(ENABLE_SIMD_WITH_AVX)
add_compile_options(-mavx)

add_executable(${PROJECT_NAME} ${SRC} ../main.cc)
target_link_libraries(${PROJECT_NAME} "-lgtest")
