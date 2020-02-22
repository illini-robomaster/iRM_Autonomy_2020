project(communication)

set(SOURCES serial.cc crc.cc protocol.cc)

add_library(communication ${SOURCES})
