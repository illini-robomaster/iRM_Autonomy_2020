/**
 * @file example_hello.cc
 * @brief example usage of using boost python to wrap generic C++ functions
 * @author Alvin Sun
 * @date 2019-11-17
 */

#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

const std::string greet() {
  return "Hello World!";
}

PYBIND11_MODULE(cc_example_hello, m) {
  m.def("greet", greet);
}
