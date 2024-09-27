#pragma once

#include "simd/config/inline.h"

#include <cstring>
#include <cstdint>

namespace simd {
namespace bits {
template <typename TO, typename FROM>
SIMD_INLINE
TO cast(FROM from)
{
    static_assert(sizeof(FROM) == sizeof(TO),
        "bit cast from `FROM` to `TO` must be same size");
    TO to{};
    std::memcpy(&to, &from, sizeof(to));
    return to;
}

/// make one T value with bits pattern:
/// 111....111
template <typename T>
SIMD_INLINE
T ones()
{
    union {
        char b[sizeof(T)];
        T t;
    };
    #pragma unroll
    for (int i = 0; i < sizeof(T); i++) {
        b[i] = 0xFF;
    }
    return t;
}

/// make one T value with bits pattern:
/// 0000....000
template <typename T>
SIMD_INLINE
T zeros()
{
    return T{};
}

/// make one T value with bits pattern:
/// 1000....000
template <typename T>
SIMD_INLINE
T one_zeros()
{
    union {
        char b[sizeof(T)];
        T t;
    };
    t = T{};
    b[sizeof(T) - 1] |= 0x80;
    return t;
}

template <typename T>
SIMD_INLINE
T signmask()
{
    return one_zeros<T>();
}

/// return bit state at msb: true for 1, false for 0
template <typename T>
SIMD_INLINE
bool at_msb(T d)
{
    union {
        char b[sizeof(T)];
        T t;
    };
    t = d;
    return (b[sizeof(T) - 1] & 0x80) != 0;
}

SIMD_INLINE
int count1(uint64_t x)
{
    int cnt = 0;
    while (x) {
        cnt++;
        x &= (x - 1);
    }
    return cnt;
}
}  // namespace bits

template <typename TO, typename FROM>
SIMD_INLINE
TO bits_cast(FROM from)
{
    return bits::cast<TO>(from);
}

}  // namespace simd
