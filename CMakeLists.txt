cmake_minimum_required(VERSION 3.12)

project(iRM CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 14)  # use c++14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra") # low warning tolerance
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG") # add debug flag for debug build

include(cmake/find_boost_python.cmake)
include(cmake/irm_cmake_utils.cmake)

include_directories(.)

enable_testing()
include(GoogleTest)

add_subdirectory(third_party)
add_subdirectory(utils)
