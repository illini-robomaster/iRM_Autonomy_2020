/**
 * @file example_class.cc
 * @brief example usage of using boost python to wrap generic C++ classes
 * @author Alvin Sun
 * @date 2019-11-17
 */
#include <string>

#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

class WrappedClass {
 public:
  WrappedClass() : value_(0), name_("") {}
  WrappedClass(int value) : value_(value), name_("") {}
  WrappedClass(const std::string& name): value_(0), name_(name) {}
  WrappedClass(int value, const std::string& name) : value_(value), name_(name) {}

  int get_value() {
    return value_;
  }

  void set_value(int value) {
    value_ = value;
  }

  const std::string get_name() {
    return name_;
  }

  void set_name(const std::string& name) {
    name_ = name;
  }

  WrappedClass operator+=(const WrappedClass &other) {
    value_ += other.value_;
    name_ += other.name_;
    return *this;
  }

  WrappedClass operator+(const WrappedClass &other) {
    return WrappedClass(value_ + other.value_, name_ + other.name_);
  }

 private:
  int value_;
  std::string name_;
};

BOOST_PYTHON_MODULE(cc_example_class) {
  class_<WrappedClass>("WrappedClass")
    // you can overload different class constructors
    .def(init<int>())
    .def(init<const std::string&>())
    .def(init<int, const std::string&>())
    // hold school getter / setter functions wrapper
    .def("get_value", &WrappedClass::get_value)
    .def("set_value", &WrappedClass::set_value)
    // python class property wrapper
    .add_property("name", &WrappedClass::get_name, &WrappedClass::set_name)
    // operator overloading
    .def(self + self)
    .def(self += self);
}
