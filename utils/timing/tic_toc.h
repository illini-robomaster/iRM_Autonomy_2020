#pragma once

#include <chrono>

namespace timing {

using namespace std;

class TicToc {
 public:
  TicToc(bool quiet = false);
  
  void Tic();
  void Toc();
 
 private:
  chrono::time_point<chrono::high_resolution_clock> start_;
  bool quiet_;
};

} // namespace timing
