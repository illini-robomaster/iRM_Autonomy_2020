#include <string>

#include <boost/python.hpp>

using namespace boost::python;

const std::string greet() {
  return "hello, world";
}

BOOST_PYTHON_MODULE(cc_example_hello) {
  def("greet", greet);
}
