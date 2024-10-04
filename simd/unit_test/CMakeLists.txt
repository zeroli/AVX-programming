cmake_minimum_required(VERSION 3.17)

add_subdirectory(sse)
add_subdirectory(fma3_sse)
add_subdirectory(avx)
add_subdirectory(fma3_avx)
add_subdirectory(avx2)
add_subdirectory(fma3_avx2)
add_subdirectory(avx512)

project(simd_ut CXX)

aux_source_directory(. SRC)
aux_source_directory(generic/ SRC)
add_compile_options(-msse4.2)
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} "-lgtest")
