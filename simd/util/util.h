#pragma once

#include "simd/config/config.h"

#include <sstream>

namespace simd {
namespace util {
struct ArchRegConfigTag { };

inline std::ostream& operator <<(std::ostream& os, ArchRegConfigTag) noexcept
{
    os << "SIMD Arch/Reg Configuration:" << "\n";
    os << "SIMD_WITH_AVX512:    " << SIMD_WITH_AVX512 << "\n";
    os << "SIMD_WITH_FMA3_AVX2: " << SIMD_WITH_FMA3_AVX2 << "\n";
    os << "SIMD_WITH_AVX2:      " << SIMD_WITH_AVX2 << "\n";
    os << "SIMD_WITH_FMA3_AVX:  " << SIMD_WITH_FMA3_AVX << "\n";
    os << "SIMD_WITH_AVX:       " << SIMD_WITH_AVX << "\n";
    os << "SIMD_WITH_FMA3_SSE:  " << SIMD_WITH_FMA3_SSE << "\n";
    os << "SIMD_WITH_SSE:       " << SIMD_WITH_SSE << "\n";
    return os;
}

inline std::string DumpArchRegConfig()
{
    std::ostringstream os;
    os << ArchRegConfigTag();
    return os.str();
}

inline void DumpArchRegConfig(std::ostream& os)
{
    os << ArchRegConfigTag();
}

/// fetch the arch/reg name for one vector
/// one you have one Vec<T, W>, either object or just type
/// ask its underlying arch/reg name through this API:
/// `std::cout << arch_name<Vec<int, 4>>();`
/// it's compile-time query
/// same behavior as `Vec<T, W>::arch_name()`
template <typename VT>
inline constexpr const char* arch_name(const VT* vec = nullptr)
{
    using arch = typename VT::arch_t;
    return arch::name();
}
}  // namespace util
}  // namespace simd
