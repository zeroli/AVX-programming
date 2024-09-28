cmake_minimum_required(VERSION 3.17)

project(simd_ut_avx512 CXX)

aux_source_directory(. SRC)

add_compile_definitions(ENABLE_SIMD_WITH_AVX512)
add_compile_options(-mavx -mavx2 -mavx512f)

add_executable(${PROJECT_NAME} ${SRC} ../main.cc)
target_link_libraries(${PROJECT_NAME} "-lgtest")
