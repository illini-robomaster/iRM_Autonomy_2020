#include <gtest/gtest.h>

#define TEST_FM(class_name, method_name, ...) \
  TEST_F(class_name, method_name) {           \
    method_name(__VA_ARGS__);                 \
  }

class TestBase : public testing::Test {
 public:
  explicit TestBase();
  virtual ~TestBase();
};
