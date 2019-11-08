# helper function for adding cpp tests
#   will look into the "./tests" directory and search for <test_name>.cc
#   example usage:
#       irm_add_cc_test(NAME random_test DEPENDS gtest_util)
function(irm_add_cc_test)
    cmake_parse_arguments(TEST_ARG "" "NAME" "DEPENDS" ${ARGN})
    add_executable(${TEST_ARG_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${TEST_ARG_NAME}.cc)
    set_target_properties(${TEST_ARG_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
    target_link_libraries(${TEST_ARG_NAME} ${TEST_ARG_DEPENDS})
    gtest_discover_tests(${TEST_ARG_NAME})
endfunction()

# helper function for generate static librareis from lcmtypes
#   will look into the "./lcmtypes" directory and search for <lcm_name>.lcm
#   example usage:
#       irm_add_lcm_library(NAME example LIB example_lcm)
#       the ${example_lcm} will be set to the generated library from example.lcm structures
function(irm_add_lcm_library)
    cmake_parse_arguments(LCM "" "NAME;LIB" "" ${ARGN})
    lcm_wrap_types(CPP_HEADERS LCM_HEADERS CPP11
                   DESTINATION ${CMAKE_BINARY_DIR}/lcmtypes
                   ${CMAKE_CURRENT_SOURCE_DIR}/lcmtypes/${LCM_NAME}.lcm)
    lcm_add_library(${LCM_NAME}_lcm CPP ${LCM_HEADERS})
    target_include_directories(${LCM_NAME}_lcm INTERFACE ${CMAKE_BINARY_DIR})
    set(${LCM_LIB} ${LCM_NAME}_lcm PARENT_SCOPE)
endfunction()
