add_subdirectory(timing)
add_subdirectory(gtest_utils)
add_subdirectory(lcm_utils)

irm_add_lcm_library(NAME example LIB example_lcm)

irm_add_cc_test(NAME timing_test DEPENDS gtest_utils timing)
irm_add_cc_test(NAME lcm_utils_test DEPENDS gtest_utils ${example_lcm} lcm_utils)

add_executable(lcm_file_logger executables/lcm_file_logger.cc)
target_link_libraries(lcm_file_logger lcm_utils)
