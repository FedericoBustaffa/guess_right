cmake_minimum_required(VERSION 3.15...4.0)

project(guess_right LANGUAGES CXX)

find_package (Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "-Wall -pedantic")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Using pybind11 found at: ${pybind11_DIR}")

file(GLOB_RECURSE HEADERS "include/*.hpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(guess_right STATIC ${HEADERS} ${SOURCES})
target_include_directories(guess_right PUBLIC include)
target_link_libraries(guess_right)

pybind11_add_module(core MODULE ${SOURCES} guess_right/bindings.cpp ${HEADERS})
target_include_directories(core PRIVATE include)
install(TARGETS core DESTINATION guess_right)

add_subdirectory(test)
