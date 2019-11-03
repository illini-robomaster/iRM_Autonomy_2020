#include <boost/log/trivial.hpp>

#define TRIVIAL_LOG_TRACE   BOOST_LOG_TRIVIAL(trace)
#define TRIVIAL_LOG_DEBUG   BOOST_LOG_TRIVIAL(debug)
#define TRIVIAL_LOG_INFO    BOOST_LOG_TRIVIAL(info)
#define TRIVIAL_LOG_WARNING BOOST_LOG_TRIVIAL(warning)
#define TRIVIAL_LOG_ERROR   BOOST_LOG_TRIVIAL(error)
#define TRIVIAL_LOG_FATAL   BOOST_LOG_TRIVIAL(fatal)

namespace logging {};
