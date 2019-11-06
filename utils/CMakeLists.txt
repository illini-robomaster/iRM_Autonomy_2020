add_subdirectory(timing)
add_subdirectory(gtest_utils)

#add_executable(timing_test tests/timing_test.cc)
#target_link_libraries(timing_test gtest_utils timing)
#gtest_discover_tests(timing_test)

irm_add_cc_test(NAME timing_test DEPENDS gtest_utils timing)
