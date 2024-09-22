#pragma once

#include "simd/types/sse_register.h"
#include "simd/types/vec.h"

namespace simd {
namespace kernel {
namespace generic {

using namespace types;

/// sign
template <typename T, size_t W>
struct sign<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        using A = typename Vec<T, W>::arch_t;
        using vec_t = Vec<T, W>;
        // +1 for positive, -1 for negative, 0 for zero
        vec_t ret = kernel::select(x > 0, vec_t(1), vec_t(0), A{})
                  - kernel::select(x < 0, vec_t(1), vec_t(0), A{});
        return ret;
    }
};

template <typename T, size_t W>
struct sign<T, W, REQUIRE_FLOATING(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        using A = typename Vec<T, W>::arch_t;
        using vec_t = Vec<T, W>;
        // +1 for positive, -1 for negative, 0 for zero
        vec_t ret = kernel::select(x > 0, vec_t(1), vec_t(0), A{})
                  - kernel::select(x < 0, vec_t(1), vec_t(0), A{});
        return ret;
    }
};

/// bitofsign
template <typename T, size_t W>
struct bitofsign<T, W, REQUIRE_INTEGRAL(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        static_check_supported_type<T, 8>();

        using vec_t = Vec<T, W>;
        if (std::is_unsigned<T>::value) {
            return vec_t(0);
        } else {
            return x & vec_t(bits::signmask<T>());
        }
    }
};

template <typename T, size_t W>
struct bitofsign<T, W, REQUIRE_FLOATING(T)>
{
    static Vec<T, W> apply(const Vec<T, W>& x) noexcept
    {
        using vec_t = Vec<T, W>;
        vec_t ret = x & constants::signmask<vec_t>();
        return ret;
    }
};
}  // namespace generic
}  // namespace kernel
}  // namespace simd
