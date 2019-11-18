project(timing)

set(SOURCES tic_toc.cc tic_toc.h)

irm_add_lcm_library(tic_toc_lcm SOURCES tic_toc.lcm)

add_library(timing ${SOURCES})
target_link_libraries(timing ${tic_toc_lcm})
