#include <iostream>

#include "utils/lcm_utils/logging.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <output_log_file>" << std::endl;
    return 1;
  }

  lcm::LCMFileLogger logger(argv[1]);

  logger.Start();
  std::cin.get();
  logger.Stop();

  return 0;
}
