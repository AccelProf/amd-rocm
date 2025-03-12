#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>

#include "defines.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace common
{
using callback_name_info = rocprofiler::sdk::callback_name_info;
using buffer_name_info   = rocprofiler::sdk::buffer_name_info;

inline auto
get_buffer_tracing_names()
{
    return rocprofiler::sdk::get_buffer_tracing_names();
}

inline auto
get_callback_tracing_names()
{
    return rocprofiler::sdk::get_callback_tracing_names();
}
}  // namespace common
