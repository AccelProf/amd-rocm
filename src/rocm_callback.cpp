#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <mutex>
#include <vector>
#include <iostream>
#include <iomanip>

#include "sanalyzer.h"
#if AMDROCM_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif

#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y)         ROCPROFILER_VAR_NAME_COMBINE(X, Y)
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


namespace rocm_accelprof {

rocprofiler_client_id_t* client_id = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t client_ctx = {0};

void tool_tracing_ctrl_callback(rocprofiler_callback_tracing_record_t record,
                                rocprofiler_user_data_t*,
                                void* client_data)
{
    auto* ctx = static_cast<rocprofiler_context_id_t*>(client_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause) {
        ROCPROFILER_CALL(rocprofiler_stop_context(*ctx), "pausing client context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume) {
        ROCPROFILER_CALL(rocprofiler_start_context(*ctx), "resuming client context");
    }
}

void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t* user_data,
                           void* callback_data)
{
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) return;

    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        const char* _operation = nullptr;
        rocprofiler_query_callback_tracing_kind_operation_name(record.kind, record.operation, &_operation, nullptr);
        PRINT("[ROCMPROF INFO] Event: %s\n", _operation);
    }

    if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
        const char* _operation = nullptr;
        rocprofiler_query_callback_tracing_kind_operation_name(record.kind, record.operation, &_operation, nullptr);
        PRINT("[ROCMPROF INFO] Event: %s\n", _operation);
    }

    if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc) {
        void* raw_ptr = nullptr;
        uint64_t size = 0;
        auto cb = [](rocprofiler_callback_tracing_kind_t,
            rocprofiler_tracing_operation_t,
            uint32_t arg_num,
            const void* const arg_value_addr,
            int32_t,
            const char* arg_type,
            const char* arg_name,
            const char*,
            int32_t,
            void* cb_data) -> int {
            auto* pair = static_cast<std::pair<void**, uint64_t*>*>(cb_data);
            if (std::string(arg_name) == "ptr") {
                *pair->first = *static_cast<void* const*>(arg_value_addr);
            } else if (std::string(arg_name) == "size") {
                *pair->second = *static_cast<const uint64_t*>(arg_value_addr);
            }

            return 0;
        };
        std::pair<void**, uint64_t*> out{&raw_ptr, &size};
        ROCPROFILER_CALL(
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, cb, /*max_deref=*/1, &out),
            "failed to iterate hipMalloc arguments");

        // std::cerr << "hipMalloc: ptr=" << raw_ptr << ", size=" << size << "\n";
        PRINT("[ROCMPROF INFO] hipMalloc: ptr=%p, size=%zu\n", raw_ptr, size);
        yosemite_alloc_callback((uint64_t)raw_ptr, size, 0);
    } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipFree) {
        void* ptr = nullptr;
        auto cb = [](rocprofiler_callback_tracing_kind_t,
                    rocprofiler_tracing_operation_t,
                    uint32_t,
                    const void* const arg_value_addr,
                    int32_t,
                    const char*,
                    const char* arg_name,
                    const char*,
                    int32_t,
                    void* cb_data) -> int {
            if (std::string(arg_name) == "ptr") {
                *static_cast<void**>(cb_data) = *static_cast<void* const*>(arg_value_addr);
            }
            return 0;
        };
        ROCPROFILER_CALL(
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, cb, 1, &ptr),
            "failed to iterate hipFree arguments");

        // std::cerr << "hipFree: ptr=" << ptr << "\n";
        PRINT("[ROCMPROF INFO] hipFree: ptr=%p\n", ptr);
        yosemite_free_callback((uint64_t)ptr, 0, 0);
    } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy) {
        void* dst = nullptr;
        const void* src = nullptr;
        size_t size = 0;
        struct memcpy_args {
            void** dst;
            const void** src;
            size_t* size;
        };
        memcpy_args out{&dst, &src, &size};
        auto cb = [](rocprofiler_callback_tracing_kind_t,
                    rocprofiler_tracing_operation_t,
                    uint32_t,
                    const void* const arg_value_addr,
                    int32_t,
                    const char*,
                    const char* arg_name,
                    const char*,
                    int32_t,
                    void* cb_data) -> int {
            auto* p = static_cast<memcpy_args*>(cb_data);
            if (std::string(arg_name) == "dst") {
                *p->dst = *static_cast<void* const*>(arg_value_addr);
            } else if (std::string(arg_name) == "src") {
                *p->src = *static_cast<void* const*>(arg_value_addr);
            } else if (std::string(arg_name) == "size") {
                *p->size = *static_cast<const size_t*>(arg_value_addr);
            }
            return 0;
        };

        ROCPROFILER_CALL(
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, cb, 1, &out),
            "failed to iterate hipMemcpy arguments");

        // std::cerr << "hipMemcpy: dst=" << dst << ", src=" << src << ", size=" << size << "\n";
        PRINT("[ROCMPROF INFO] hipMemcpy: dst=%p, src=%p, size=%zu\n", dst, src, size);
        yosemite_memcpy_callback((uint64_t)dst, (uint64_t)src, size, false, 0);
    } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset) {
        void* ptr = nullptr;
        int value = 0;
        size_t size = 0;

        struct memset_args {
            void** ptr;
            int* value;
            size_t* size;
        };
        memset_args out{&ptr, &value, &size};

        auto cb = [](rocprofiler_callback_tracing_kind_t,
                    rocprofiler_tracing_operation_t,
                    uint32_t,
                    const void* const arg_value_addr,
                    int32_t,
                    const char*,
                    const char* arg_name,
                    const char*,
                    int32_t,
                    void* cb_data) -> int {
            auto* p = static_cast<memset_args*>(cb_data);
            if (std::string(arg_name) == "dst") {
                *p->ptr = *static_cast<void* const*>(arg_value_addr);
            } else if (std::string(arg_name) == "value") {
                *p->value = *static_cast<const int*>(arg_value_addr);
            } else if (std::string(arg_name) == "size") {
                *p->size = *static_cast<const size_t*>(arg_value_addr);
            }
            return 0;
        };

        ROCPROFILER_CALL(
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, cb, 1, &out),
            "failed to iterate hipMemset arguments");

        // std::cerr << "hipMemset: ptr=" << ptr << ", value=" << value << ", size=" << size << "\n";
        PRINT("[ROCMPROF INFO] hipMemset: ptr=%p, value=%d, size=%zu\n", ptr, value, size);
        yosemite_memset_callback((uint64_t)ptr, size, value, false);
    } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel) {
        const void* func_ptr = nullptr;
        dim3 grid_dim = {};
        dim3 block_dim = {};
        void* stream = nullptr;
        uint32_t shared_mem = 0;

        struct kernel_args {
            const void** func_ptr;
            dim3* grid_dim;
            dim3* block_dim;
            uint32_t* shared_mem;
            void** stream;
        };
        
        kernel_args out{&func_ptr, &grid_dim, &block_dim, &shared_mem, &stream};
        auto cb = [](rocprofiler_callback_tracing_kind_t,
                    rocprofiler_tracing_operation_t,
                    uint32_t,
                    const void* const arg_value_addr,
                    int32_t,
                    const char*,
                    const char* arg_name,
                    const char*,
                    int32_t,
                    void* cb_data) -> int {
            auto* p = static_cast<kernel_args*>(cb_data);
            if (std::string(arg_name) == "func") {
                *p->func_ptr = *static_cast<const void* const*>(arg_value_addr);
            } else if (std::string(arg_name) == "gridDim") {
                *p->grid_dim = *static_cast<const dim3*>(arg_value_addr);
            } else if (std::string(arg_name) == "blockDim") {
                *p->block_dim = *static_cast<const dim3*>(arg_value_addr);
            } else if (std::string(arg_name) == "sharedMemBytes") {
                *p->shared_mem = *static_cast<const uint32_t*>(arg_value_addr);
            } else if (std::string(arg_name) == "stream") {
                *p->stream = *static_cast<void* const*>(arg_value_addr);
            }
            return 0;
        };

        ROCPROFILER_CALL(
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, cb, 1, &out),
            "failed to iterate hipLaunchKernel arguments");

        // std::cerr << "hipLaunchKernel: func=" << func_ptr
        //     << ", grid=(" << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z << ")"
        //     << ", block=(" << block_dim.x << "," << block_dim.y << "," << block_dim.z << ")"
        //     << ", sharedMem=" << shared_mem
        //     << ", stream=" << stream << "\n";

        PRINT("[ROCMPROF INFO] hipLaunchKernel: func=%p, grid=(%d,%d,%d), block=(%d,%d,%d), sharedMem=%d, stream=%p\n",
            func_ptr, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z, shared_mem, stream);
        std::stringstream ss;
        ss << func_ptr;
        yosemite_kernel_start_callback(ss.str());
    }

    auto info = std::stringstream{};
    info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
         << record.correlation_id.internal << ", kind=" << record.kind
         << ", operation=" << std::setw(3) << record.operation << ", phase=" << record.phase;

    auto info_data_cb = [](rocprofiler_callback_tracing_kind_t,
                           rocprofiler_tracing_operation_t,
                           uint32_t          arg_num,
                           const void* const arg_value_addr,
                           int32_t           indirection_count,
                           const char*       arg_type,
                           const char*       arg_name,
                           const char*       arg_value_str,
                           int32_t           dereference_count,
                           void*             cb_data) -> int {
        auto& dss = *static_cast<std::stringstream*>(cb_data);
        dss << ((arg_num == 0) ? "(" : ", ");
        dss << arg_num << ": " << arg_name << "=" << arg_value_str;
        (void) arg_value_addr;
        (void) arg_type;
        (void) indirection_count;
        (void) dereference_count;

        PRINT("[ROCMPROF INFO] Event: %s\n", dss.str().c_str());
        return 0;
    };

    int32_t max_deref = (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) ? 1 : 2;
    auto    info_data = std::stringstream{};
    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                         record, info_data_cb, max_deref, static_cast<void*>(&info_data)),
                     "Failure iterating trace operation args");

    auto info_data_str = info_data.str();
    if(!info_data_str.empty()) info << " " << info_data_str << ")";

    PRINT("%s\n", info.str().c_str());
}

void tool_control_init(rocprofiler_context_id_t& primary_ctx)
{
    // Create a specialized (throw-away) context for handling ROCTx profiler pause and resume.
    // A separate context is used because if the context that is associated with roctxProfilerPause
    // disabled that same context, a call to roctxProfilerResume would be ignored because the
    // context that enables the callback for that API call is disabled.
    auto cntrl_ctx = rocprofiler_context_id_t{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&cntrl_ctx), "control context creation failed");

    // enable callback marker tracing with only the pause/resume operations
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         cntrl_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         tool_tracing_ctrl_callback,
                         &primary_ctx),
                     "callback tracing service failed to configure");

    // start the context so that it is always active
    ROCPROFILER_CALL(rocprofiler_start_context(cntrl_ctx), "start of control context");
}

int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data) {
    client_fini_func = fini_func;
    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    // enable the control
    tool_control_init(client_ctx);

    rocprofiler_callback_tracing_kind_t kinds[] = {
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
    };

    for(auto itr : kinds)
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                        client_ctx, itr, nullptr, 0, tool_tracing_callback, tool_data),
                        "callback tracing service failed to configure");
    }
    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "failure checking context validity");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
    return 0;
}

void tool_fini(void*) {}

} // namespace

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t version,
                                                                    const char* runtime_version,
                                                                    uint32_t priority,
                                                                    rocprofiler_client_id_t* id) {
    id->name = "AccelProf";
    rocm_accelprof::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";
    printf("%s\n", info.str().c_str());

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &rocm_accelprof::tool_init,
                                            &rocm_accelprof::tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
