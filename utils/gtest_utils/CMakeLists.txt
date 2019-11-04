project(gtest_utils)

set(SOURCES test_base.h test_base.cc)

add_library(gtest_utils ${SOURCES})
target_include_directories(gtest_utils PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(gtest_utils PUBLIC ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
