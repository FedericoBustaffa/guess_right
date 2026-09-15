cmake_minimum_required(VERSION 4.3)

project(mlfs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB_RECURSE HEADERS "include/*.hpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(mlfs STATIC ${HEADERS} ${SOURCES})
target_include_directories(mlfs PUBLIC "include/")
target_link_libraries(mlfs)

add_subdirectory(test)
add_subdirectory(examples)
