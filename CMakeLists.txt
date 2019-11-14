cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)  # use c++14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra")
set(CMAKE_CXX_COMPILER g++)

project(iRM CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(cmake/irm_cmake_utils.cmake)

option(DEBUG "set to ON when debugging stuff" OFF)

if (DEBUG)
    add_compile_options(-g)
    add_definitions(-DDEBUG)
else ()
    add_compile_options(-O3)
    add_definitions(-DNDEBUG)
endif ()

SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED)
find_package(lcm REQUIRED)
find_package(GTest REQUIRED)

include(${LCM_USE_FILE}) # lcm related cmake helper functions

include_directories(.)

enable_testing()

add_subdirectory(utils)
