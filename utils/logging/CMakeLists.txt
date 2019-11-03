project(logging)

set(SOURCES trivial.h trivial.cc)

add_library(logging ${SOURCES})
target_link_libraries(logging PUBLIC ${Boost_LOG_LIBRARY})
target_include_directories(logging PUBLIC ${Boost_INCLUDE_DIRS})
