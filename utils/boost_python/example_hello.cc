/**
 * @file example_hello.cc
 * @brief example usage of using boost python to wrap generic C++ functions
 * @author Alvin Sun 
 * @date 2019-11-17
 */

#include <string>

#include <boost/python.hpp>

using namespace boost::python;

const std::string greet() {
  return "Hello World!";
}

BOOST_PYTHON_MODULE(cc_example_hello) {
  def("greet", greet);
}
