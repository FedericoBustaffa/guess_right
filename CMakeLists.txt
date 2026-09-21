cmake_minimum_required(VERSION 3.15...4.0)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE HEADERS "include/*.hpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")

# build static C++ library
add_library(guess_right STATIC ${HEADERS} ${SOURCES})
target_include_directories(guess_right PUBLIC include)

# build python module
pybind11_add_module(core MODULE ${SOURCES} guess_right/bindings.cpp ${HEADERS})
target_include_directories(core PRIVATE include)
install(TARGETS core DESTINATION guess_right)

add_subdirectory(test)
