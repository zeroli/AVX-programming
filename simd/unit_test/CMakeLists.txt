cmake_minimum_required(VERSION 3.17)

project(simd_ut CXX)

aux_source_directory(. SRC)
aux_source_directory(./sse SRC)
aux_source_directory(./generic SRC)

include_directories(../include)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} "-lgtest")
