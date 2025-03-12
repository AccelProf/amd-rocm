#pragma once

#if !defined(ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM)
#    if defined __has_include
#        if __has_include(<ghc/filesystem.hpp>)
#            define ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM 1
#        else
#            define ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM 0
#        endif
#    else
#        define ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM 0
#    endif
#endif

#if ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM == 0
#    if defined __has_include
#        if __has_include(<version>)
#            include <version>
#        endif
#    endif

#    if defined(__cpp_lib_filesystem)
#        define ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM 1
#    else
#        if defined __has_include
#            if __has_include(<filesystem>)
#                define ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM 1
#            endif
#        endif
#    endif
#endif

// include the correct filesystem header
#if defined(ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM) &&                                         \
    ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM > 0
#    include <ghc/filesystem.hpp>
#elif defined(ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM) &&                                       \
    ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM > 0
#    include <filesystem>
#else
#    include <experimental/filesystem>
#endif

namespace common
{
#if defined(ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM) &&                                         \
    ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM > 0
namespace fs = ::ghc::filesystem;  // NOLINT(misc-unused-alias-decls)
#elif defined(ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM) &&                                       \
    ROCPROFILER_SAMPLES_HAS_CPP_LIB_FILESYSTEM > 0
namespace fs = ::std::filesystem;  // NOLINT(misc-unused-alias-decls)
#else
namespace fs = ::std::experimental::filesystem;  // NOLINT(misc-unused-alias-decls)
#endif
}  // namespace common
