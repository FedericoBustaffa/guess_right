cmake_minimum_required(VERSION 4.3)

project(guess_right)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

file(GLOB_RECURSE HEADERS "include/*.hpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(guess_right STATIC ${HEADERS} ${SOURCES})
target_include_directories(guess_right PUBLIC "include/")
target_link_libraries(guess_right)

add_subdirectory(test)
