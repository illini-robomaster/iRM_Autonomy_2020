project(gtest_utils)

set(SOURCES test_base.h test_base.cc)

add_library(gtest_utils ${SOURCES})
target_link_libraries(gtest_utils PUBLIC gtest gtest_main)

