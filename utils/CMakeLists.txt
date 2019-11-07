add_subdirectory(timing)
add_subdirectory(gtest_utils)

irm_add_cc_test(NAME timing_test DEPENDS gtest_utils timing)
