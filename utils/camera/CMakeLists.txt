project(camera)

set(SOURCES camera_driver.cc)

add_library(camera ${SOURCES})

target_link_libraries(camera ${OpenCV_LIBS} pthread)
