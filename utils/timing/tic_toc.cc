#include "utils/logging/trivial.h"
#include "utils/timing/tic_toc.h"

namespace timing {

TicToc::TicToc(bool quiet) : quiet_(quiet) {
  Tic();
}

void TicToc::Tic() {
  start_ = chrono::high_resolution_clock::now();
}

void TicToc::Toc() {
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> duration = start_ - end;
  if (!quiet_) {
    TRIVIAL_LOG_INFO << "Time elapsed: " << duration.count() << std::endl;
  }
}

} // namespace timing
