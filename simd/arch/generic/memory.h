#pragma once

namespace simd { namespace kernel { namespace generic {
using namespace types;

template <typename T, size_t W>
struct load_aligned<std::complex<T>, W>
{
    using value_type = std::complex<T>;
    SIMD_INLINE
    static Vec<value_type, W> apply(const value_type* mem) noexcept
    {
        using vec_t = Vec<T, W>;
        using A = typename vec_t::arch_t;
        auto vlo = vec_t::load_aligned((const T*)mem);
        auto vhi = vec_t::load_aligned((const T*)mem + W);
        Vec<value_type, W> ret = kernel::load_complex(vlo, vhi, A{});
        return ret;
    }
};

template <typename T, size_t W>
struct load_unaligned<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static Vec<value_type, W> apply(const value_type* mem) noexcept
    {
        using vec_t = Vec<T, W>;
        using A = typename vec_t::arch_t;
        auto vlo = vec_t::load_unaligned((const T*)mem);
        auto vhi = vec_t::load_unaligned((const T*)mem + W);
        Vec<value_type, W> ret = kernel::load_complex(vlo, vhi, A{});
        return ret;
    }
};

template <typename T, size_t W>
struct store_aligned<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static void apply(value_type* mem, const Vec<value_type, W>& x) noexcept
    {
        using vec_t = Vec<T, W>;
        using A = typename vec_t::arch_t;
        auto vlo = kernel::complex_packlo(x.real(), x.imag(), A{});
        auto vhi = kernel::complex_packhi(x.real(), x.imag(), A{});
        vlo.store((T*)mem);
        vhi.store((T*)mem + W);
    }
};

template <typename T, size_t W>
struct store_unaligned<std::complex<T>, W>
{
    using value_type = std::complex<T>;

    SIMD_INLINE
    static void apply(value_type* mem, const Vec<value_type, W>& x) noexcept
    {
        using vec_t = Vec<T, W>;
        using A = typename vec_t::arch_t;
        auto vlo = kernel::complex_packlo(x.real(), x.imag(), A{});
        auto vhi = kernel::complex_packhi(x.real(), x.imag(), A{});
        vlo.storeu((T*)mem);
        vhi.storeu((T*)mem + W);
    }
};
} } } // namespace simd::kernel::generic
