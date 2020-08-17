cmake_minimum_required(VERSION 3.1)

project(lcm)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(CMAKE_VERSION VERSION_LESS 3.7)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3.7")
endif()

find_package(GLib2 REQUIRED)

# Configuration and utility functions
include(lcm-cmake/config.cmake NO_POLICY_SCOPE)
include(lcm-cmake/functions.cmake)
include(lcm-cmake/version.cmake)
include(lcm-cmake/lcmUtilities.cmake)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if (WIN32)
  add_definitions(-DWIN32 -D_CRT_SECURE_NO_WARNINGS)
  include_directories(${lcm_SOURCE_DIR}/WinSpecific/include)
  include_directories(${lcm_SOURCE_DIR}/WinSpecific/getopt)
  include_directories(${lcm_SOURCE_DIR})
  add_subdirectory(WinSpecific)
  set(lcm-winport lcm-winport)
  get_filename_component(LCM_LCMGEN_PATH ${GLIB2_GLIB_RUNTIME} DIRECTORY)
  set(LCM_USE_GLIB_RUNTIME "set(LCM_LCMGEN_PATH \"${LCM_LCMGEN_PATH}\")")
else()
  set(lcm-winport)
endif()

# Core modules
add_subdirectory(lcm)
add_subdirectory(lcmgen)
add_subdirectory(lcm-logger)

# Documentation (Main, C/C++, .NET)
add_subdirectory(docs)

# Python
lcm_option(
  LCM_ENABLE_PYTHON
  "Build Python bindings and utilities"
  PYTHON_FOUND Python)
if(LCM_ENABLE_PYTHON)
  add_subdirectory(lcm-python)
endif()

