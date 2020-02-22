#include "utils/communication/crc.h"
#include "utils/communication/protocol.h"

#define SIZEOF_COMMAND_ID 2

#define TX2_SOF           0xA0

namespace communication {

size_t get_packet_size(uint16_t data_length) {
  return sizeof(header_t) + SIZEOF_CRC8 + SIZEOF_COMMAND_ID + data_length + SIZEOF_CRC16;
}

std::vector<uint8_t> encode(uint16_t command_id, void *data, uint16_t length) {
  // create memory for the whole data packet
  const size_t packet_size = get_packet_size(length);
  std::vector<uint8_t> encoded(packet_size);
  uint8_t *raw_ptr = encoded.data();

  // populate header
  header_t *header = reinterpret_cast<header_t*>(raw_ptr);
  header->sof = TX2_SOF;
  header->data_length = length;
  // TODO(alvin): utilize sequence id in the future, or remove this field
  header->sequence_id = 0;
  append_crc8(raw_ptr, sizeof(header_t));

  // populate message body
  raw_ptr += sizeof(header_t) + SIZEOF_CRC8;
  *(uint16_t*)raw_ptr = command_id;
  raw_ptr += SIZEOF_COMMAND_ID;
  memcpy(raw_ptr, data, length);
  append_crc16(encoded.data(), packet_size - SIZEOF_CRC16);

  return encoded;
}

header_t* decode_header(uint8_t *data) {
  if (check_crc8(data, sizeof(header_t) + SIZEOF_CRC8))
    return reinterpret_cast<header_t*>(data);
  else
    return NULL;
}

body_t* decode_body(uint8_t *data, uint16_t data_length) {
  if (check_crc16(data, get_packet_size(data_length)))
    return reinterpret_cast<body_t*>(data + sizeof(header_t) + SIZEOF_CRC8);
  else
    return NULL;
}

} /* namespace communication */
