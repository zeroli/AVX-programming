cmake_minimum_required(VERSION 3.17)

project(Ch02_01 CXX)

aux_source_directory(. SRC)

include_directories(../include)

add_executable(code ${SRC})
