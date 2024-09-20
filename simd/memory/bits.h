#pragma once

#include <cstring>

namespace simd {
namespace bits {
template <typename TO, typename FROM>
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
T zeros()
{
    return T{};
}

/// make one T value with bits pattern:
/// 1000....000
template <typename T>
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
}  // namespace bits

template <typename TO, typename FROM>
TO bits_cast(FROM from)
{
    return bits::cast<TO>(from);
}

}  // namespace simd
