#pragma once

/// avx2 could use avx implementation detail
#include "simd/arch/avx/detail.h"

namespace simd { namespace kernel { namespace avx2 {
namespace detail {
/// import all details from avx
using namespace simd::kernel::avx::detail;

}  // namespace detail
}}}  // namespace simd::kernel::avx2
