# helper function for adding cpp tests
#   will look in to the "./tests" directory and search for <test_name>.cc
#   example usage:
#       irm_add_cc_test(NAME random_test DEPENDS gtest_util)
function(irm_add_cc_test)
    cmake_parse_arguments(TEST_ARG "" "NAME" "DEPENDS" ${ARGN})
    add_executable(${TEST_ARG_NAME} tests/${TEST_ARG_NAME}.cc)    
    set_target_properties(${TEST_ARG_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
    target_link_libraries(${TEST_ARG_NAME} ${TEST_ARG_DEPENDS})
    gtest_discover_tests(${TEST_ARG_NAME})
endfunction()
