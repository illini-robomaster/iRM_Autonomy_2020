project(lcm_utils)

set(SOURCES logging.cc logging.h)

add_library(lcm_utils ${SOURCES})
target_link_libraries(lcm_utils lcm)
