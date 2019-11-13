project(timing)

set(SOURCES tic_toc.cc tic_toc.h)

irm_add_lcm_library(NAME tic_toc LIB tic_toc_lcm)

add_library(timing ${SOURCES})
target_link_libraries(timing ${tic_toc_lcm})
