add_subdirectory(timing)
add_subdirectory(gtest_utils)
add_subdirectory(lcm_utils)
add_subdirectory(communication)

irm_add_lcm_library(example_lcm SOURCES lcmtypes/example.lcm)

irm_add_cc_test(timing_test DEPENDS gtest_utils timing)
irm_add_cc_test(protocol_test DEPENDS gtest_utils communication)
if (NOT WSL)
    irm_add_cc_test(lcm_utils_test DEPENDS gtest_utils ${example_lcm} lcm_utils)
endif ()

add_executable(lcm_file_logger executables/lcm_file_logger.cc)
target_link_libraries(lcm_file_logger lcm_utils)
