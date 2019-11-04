cmake_minimum_required(VERSION 3.12)
project(iRM CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

option(DEBUG "set to ON when debugging stuff" OFF)

add_compile_options(--std=c++11)

SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED)
find_package(GTest REQUIRED)

include_directories(.)

if (DEBUG)
    add_compile_options(-g)
    add_definitions(-DDEBUG)
else ()
    add_compile_options(-O3)
    add_definitions(-DNDEBUG)
endif ()

enable_testing()
add_subdirectory(utils)
