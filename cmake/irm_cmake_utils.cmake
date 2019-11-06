# helper function for adding cpp tests
#   will look in to the "./tests" directory and search for <test_name>.cc
#   example usage:
#       irm_add_cc_test(NAME random_test DEPENDS gtest_util)
function(irm_add_cc_test)
    cmake_parse_arguments(TEST_ARG "" "NAME" "DEPENDS" ${ARGN})
    add_executable(${TEST_ARG_NAME} tests/${TEST_ARG_NAME}.cc)    
    target_link_libraries(${TEST_ARG_NAME} ${TEST_ARG_DEPENDS})
    gtest_discover_tests(${TEST_ARG_NAME})
endfunction()
