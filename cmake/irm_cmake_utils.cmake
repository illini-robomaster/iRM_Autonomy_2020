# directory that contains the generated python bindings
set(IRM_PYTHON_MODULE_DIR ${PROJECT_BINARY_DIR}/python_bindings)
# detect Windows Subsystem of Linux
if (CMAKE_SYSTEM MATCHES "Linux" AND CMAKE_SYSTEM MATCHES "Microsoft")
    set(WSL TRUE)
endif ()

## irm_add_cc_test(NAME <test_name> [DEPENDS <dep1> <dep2> ...])
#
#   helper function for adding cpp tests
#   will look into the "./tests" directory and search for <test_name>.cc
function(irm_add_cc_test test_name)
    cmake_parse_arguments(TEST_ARG "" "" "DEPENDS" ${ARGN})
    add_executable(${test_name} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_name}.cc)
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
    target_link_libraries(${test_name} ${TEST_ARG_DEPENDS})
    gtest_discover_tests(${test_name})
endfunction()

## irm_add_python_test(<test_name>)
#   helper function for adding python tests
#   will look into the "./tests" directory and search for <test_name>.py
function(irm_add_python_test test_name)
    add_test(NAME ${test_name}
             COMMAND ${Python3_EXECUTABLE} ${test_name}.py
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT
        "PYTHONPATH=${IRM_PYTHON_MODULE_DIR}:${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}")
endfunction()

## irm_add_lcm_library(<lcm_lib> SOURCES <src1>.lcm <src2>.lcm ...)
# 
#   helper function for generate static librareis from lcmtypes
#   the ${<lcm_lib>} will be set to the generated library from example.lcm structures
function(irm_add_lcm_library lcm_name)
    cmake_parse_arguments(LCM "" "" "SOURCES" ${ARGN})
    lcm_wrap_types(CPP_HEADERS LCM_HEADERS CPP11
                   DESTINATION ${CMAKE_BINARY_DIR}/lcmtypes
                   ${LCM_SOURCES})
    lcm_add_library(liblcm_${lcm_name} CPP ${LCM_HEADERS})
    target_include_directories(liblcm_${lcm_name} INTERFACE
        ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/lcmtypes)
    set(${lcm_name} liblcm_${lcm_name} PARENT_SCOPE)
endfunction()

## irm_add_python_module(cc_<module_name> 
#                        SOURCES <src1>.cc <src2>.cc ... 
#                        [DEPENDS <dep1> <dep2> ...])
#
#   helper function for generating pybind modules
#   all generated python modules will go to ${PROJECT_BINARY_DIR}/python_bindings
function(irm_add_python_module module_name)
    cmake_parse_arguments(PYBIND "" "" "SOURCES;DEPENDS" ${ARGN})
    pybind11_add_module(${module_name} ${PYBIND_SOURCES})
    target_link_libraries(${module_name} PRIVATE ${PYBIND_DEPENDS})
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${IRM_PYTHON_MODULE_DIR})
endfunction()

