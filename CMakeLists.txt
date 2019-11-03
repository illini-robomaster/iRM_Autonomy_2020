cmake_minimum_required(VERSION 3.12)
project(iRM CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

option(DEBUG "set to ON to show debug information and image plot" OFF)

add_compile_options(--std=c++11)

find_package(Boost REQUIRED)
find_package(GTest REQUIRED)

include_directories(BOOST_INCLUDE_DIRS)
include_directories(GTEST_INCLUDE_DIRS)
include_directories(.)

if (DEBUG)
    add_compile_options(-g)
    add_definitions(-DDEBUG)
else ()
    add_definitions(-DNDEBUG)
endif ()

add_subdirectory(utils)
