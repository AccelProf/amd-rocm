#pragma once
#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y)         ROCPROFILER_VAR_NAME_COMBINE(X, Y)

#define ROCPROFILER_WARN(result)                                                                   \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << #result                     \
                      << " returned error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)    \
                      << ": " << status_msg << ". This is just a warning!" << std::endl;           \
        }                                                                                          \
    }

#define ROCPROFILER_CHECK(result)                                                                  \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" << __FILE__ << ":" << __LINE__ << "] " << #result                        \
                   << " failed with error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)    \
                   << " :: " << status_msg;                                                        \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) \
                      << ": " << status_msg << std::endl;                                          \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#if HIP_VERSION >= 60300000
#    define HIP_HOST_ALLOC_FUNC hipHostMalloc
#    define HIP_HOST_FREE_FUNC  hipHostFree
#else
#    define HIP_HOST_ALLOC_FUNC hipHostMalloc
#    define HIP_HOST_FREE_FUNC  hipHostFree
#endif
