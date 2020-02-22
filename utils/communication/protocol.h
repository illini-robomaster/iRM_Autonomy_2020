#pragma once

#include <vector>
#include <cinttypes>

namespace communication {

typedef struct {
  uint8_t   sof;
  uint16_t  data_length;
  uint8_t   sequence_id;
} __attribute__((packed)) header_t;

typedef struct {
  uint16_t  command_id;
  uint8_t   data[1];
} __attribute__((packed)) body_t;

/**
 * @brief calculate full size of encoded packet given data length
 *
 * @param data_length length of the actual message data [in bytes]
 *
 * @return full size [in bytes] of the encoded packet
 */
size_t get_packet_size(uint16_t data_length);

/**
 * @brief encode data based on protocol
 *
 * @param command_id  command id 
 * @param data        start address of the data stream
 * @param length      length of data [in bytes]
 *
 * @return encoded byte array
 */
std::vector<uint8_t> encode(uint16_t command_id, void *data, uint16_t length);

/**
 * @brief decode header data
 *
 * @param data  start address of the entire encoded packet
 *
 * @return  pointer to a header struct if valid, otherwise NULL
 */
header_t* decode_header(uint8_t *data);

/**
 * @brief decode body data
 *
 * @param data        start address of the entire encoded packet
 * @param data_length length of the actual message data [in bytes]
 *
 * @return  pointer to a body struct if valid, otherwise NULL
 */
body_t* decode_body(uint8_t *data, uint16_t data_length);

} /* namespace communication */
