#include <vector>

#include "utils/communication/protocol.h"
#include "utils/gtest_utils/test_base.h"

using namespace communication;

class ProtocolTest : public TestBase {
 public:
  void EncodeDecodeTest() {
    // arbitrary data frame
    uint16_t data_length = 10;
    uint16_t command_id = 0x4242;
    uint32_t packet_size = get_packet_size(data_length);
    EXPECT_EQ(packet_size, (uint32_t)19);
    // dummy data buffer
    uint8_t data[10];
    for (size_t i = 0; i < 10; ++i)
      data[i] = i;
    // encoding
    std::vector<uint8_t> encoded_buffer = encode(command_id, data, data_length);
    // validate encoding
    //  1. header encoding
    uint8_t *raw_ptr = encoded_buffer.data();
    EXPECT_EQ(encoded_buffer.size(), packet_size);
    header_t *header;
    ASSERT_TRUE((header = decode_header(raw_ptr)));
    EXPECT_EQ(header->sof, 0xA0); // TX2 SOF;
    EXPECT_EQ(header->data_length, data_length);
    EXPECT_EQ(header->sequence_id, 0); // currently sequence id is unused
    //  2. body encoding
    body_t *body;
    ASSERT_TRUE((body = decode_body(raw_ptr, header->data_length)));
    EXPECT_EQ(body->command_id, command_id);
    for (size_t i = 0; i < 10; ++i)
      EXPECT_EQ(body->data[i], i);
  }
};

TEST_FM(ProtocolTest, EncodeDecodeTest);
