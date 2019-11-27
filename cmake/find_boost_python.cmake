# Find python interpreter + run-time libs / headers
find_package(Python REQUIRED COMPONENTS Interpreter Development)
set(Python_SUFFIX ${Python_VERSION_MAJOR}${Python_VERSION_MINOR})

if (APPLE)
    set(Boost_NO_BOOST_CMAKE "ON")
endif ()

find_package(Boost REQUIRED COMPONENTS python${Python_SUFFIX} numpy${Python_SUFFIX})

set(BOOST_PYTHON_LIB Boost::python${Python_SUFFIX})
set(BOOST_NUMPY_LIB Boost::numpy${Python_SUFFIX})

# all boost python modules are generated to this directory
set(IRM_PYTHON_MODULE_DIR ${PROJECT_BINARY_DIR}/python_bindings)
