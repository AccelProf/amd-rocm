
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/api_callback_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "rocm_callback.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "call_stack.hpp"
#include "defines.hpp"
#include "filesystem.hpp"
#include "name_info.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <ratio>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
namespace rocm_callback
{
namespace
{
using common::call_stack_t;
using common::callback_name_info;
using common::source_location;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {0};

void
print_call_stack(const call_stack_t& _call_stack)
{
    common::print_call_stack("api_callback_trace.log", _call_stack);
}

void
tool_tracing_ctrl_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t*,
                           void* client_data)
{
    auto* ctx = static_cast<rocprofiler_context_id_t*>(client_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(*ctx), "pausing client context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(*ctx), "resuming client context");
    }
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 callback_data)
{
    assert(callback_data != nullptr);

    auto     now = std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t dt  = 0;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        user_data->value = now;
    else
        dt = (now - user_data->value);

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc) {
            void* raw_ptr = nullptr;
            uint64_t size = 0;

            auto cb = [](rocprofiler_callback_tracing_kind_t,
                        rocprofiler_tracing_operation_t,
                        uint32_t          arg_num,
                        const void* const arg_value_addr,
                        int32_t,
                        const char*       arg_type,
                        const char*       arg_name,
                        const char*,
                        int32_t,
                        void*             cb_data) -> int {
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
            // yosemite_malloc_callback((uint64_t)raw_ptr, size, 0);
        }
        // memory free
        else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipFree) {
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
            // yosemite_free_callback((uint64_t)ptr, 0, 0);
        } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy) {
            void* dst = nullptr;
            const void* src = nullptr;
            size_t size = 0;

            struct {
                void** dst;
                const void** src;
                size_t* size;
            } out{&dst, &src, &size};

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
                auto* p = static_cast<decltype(out)*>(cb_data);
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

        } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset) {
            void* ptr = nullptr;
            int value = 0;
            size_t size = 0;

            struct {
                void** ptr;
                int* value;
                size_t* size;
            } out{&ptr, &value, &size};

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
                auto* p = static_cast<decltype(out)*>(cb_data);
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
        } else if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel) {
            const void* func_ptr = nullptr;
            dim3 grid_dim = {};
            dim3 block_dim = {};
            void* stream = nullptr;
            uint32_t shared_mem = 0;

            struct {
                const void** func_ptr;
                dim3* grid_dim;
                dim3* block_dim;
                uint32_t* shared_mem;
                void** stream;
            } out{&func_ptr, &grid_dim, &block_dim, &shared_mem, &stream};

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
                auto* p = static_cast<decltype(out)*>(cb_data);
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

            std::cerr << "hipLaunchKernel: func=" << func_ptr
                    << ", grid=(" << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z << ")"
                    << ", block=(" << block_dim.x << "," << block_dim.y << "," << block_dim.z << ")"
                    << ", sharedMem=" << shared_mem
                    << ", stream=" << stream << "\n";
                }
    }

    auto info = std::stringstream{};
    info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
         << record.correlation_id.internal << ", kind=" << record.kind
         << ", operation=" << std::setw(3) << record.operation << ", phase=" << record.phase
         << ", dt_nsec=" << std::setw(6) << dt;

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
        return 0;
    };

    int32_t max_deref = (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) ? 1 : 2;
    auto    info_data = std::stringstream{};
    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                         record, info_data_cb, max_deref, static_cast<void*>(&info_data)),
                     "Failure iterating trace operation args");

    auto info_data_str = info_data.str();
    if(!info_data_str.empty()) info << " " << info_data_str << ")";

    static auto _mutex = std::mutex{};
    _mutex.lock();
    static_cast<call_stack_t*>(callback_data)
        ->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
    _mutex.unlock();
}

void
tool_control_init(rocprofiler_context_id_t& primary_ctx)
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

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    callback_name_info name_info = common::get_callback_tracing_names();

    for(const auto& itr : name_info)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << itr.value << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_callback_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            std::string{itr.name}});

        for(auto [didx, ditr] : itr.items())
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << didx << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_callback_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{*ditr}});
        }
    }

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    // enable the control
    tool_control_init(client_ctx);

    for(auto itr : {ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
                    ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                    ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                    ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API})
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             client_ctx, itr, nullptr, 0, tool_tracing_callback, tool_data),
                         "callback tracing service failed to configure");
    }

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

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

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
}
}  // namespace

void
setup()
{}

void
shutdown()
{
    if(client_id) client_fini_func(*client_id);
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
}

void
stop()
{
    int status = 0;
    ROCPROFILER_CALL(rocprofiler_is_initialized(&status), "failed to retrieve init status");
    if(status != 0)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop failed");
    }
}
}  // namespace rocm_callback

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "ExampleTool";

    // store client info
    rocm_callback::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    // demonstration of alternative way to get the version info
    {
        auto version_info = std::array<uint32_t, 3>{};
        ROCPROFILER_CALL(
            rocprofiler_get_version(&version_info.at(0), &version_info.at(1), &version_info.at(2)),
            "failed to get version info");

        if(std::array<uint32_t, 3>{major, minor, patch} != version_info)
        {
            throw std::runtime_error{"version info mismatch"};
        }
    }

    // data passed around all the callbacks
    auto* client_tool_data = new std::vector<rocm_callback::source_location>{};

    // add first entry
    client_tool_data->emplace_back(
        rocm_callback::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &rocm_callback::tool_init,
                                            &rocm_callback::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
