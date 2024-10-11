#pragma once

#include "simd/types/generic_arch.h"

namespace simd {
/// AVX512 instructions
struct AVX512 : Generic
{
    static constexpr bool supported() noexcept { return SIMD_WITH_AVX512; }
    static constexpr bool available() noexcept { return true; }
    static constexpr size_t alignment() noexcept { return 64; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr const char* name() noexcept { return "AVX512"; }
};
}  // namespace simd

#if  SIMD_WITH_AVX512
#include <immintrin.h>
#include <bitset>

namespace simd {
namespace types {
using avx512_reg_i = __m512i;
using avx512_reg_f = __m512;
using avx512_reg_d = __m512d;

template <typename T, typename Enable = void>
struct avx512_reg_traits;

template <typename T>
using avx512_reg_traits_t = typename avx512_reg_traits<T>::type;

template <typename T, typename Enable = void>
struct avx512_mask_traits;

template <typename T>
using avx512_mask_traits_t = typename avx512_mask_traits<T>::type;

/// bunch of avx512 mask types for different scalar type
/// basically: sizeof(T) * mask bits * 8 = 64 * 8 = 512
/// each bit indicates one element, in compact way
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 1)> {
    using type = __mmask64;
};
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 2)> {
    using type = __mmask32;
};
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 4)> {
    using type = __mmask16;
};
template <typename T>
struct avx512_mask_traits<T, ENABLE_IF(sizeof(T) == 8)> {
    using type = __mmask8;
};

#define DECLARE_SIMD_AVX512_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
template <> \
struct avx512_reg_traits<SCALAR_TYPE> { \
    using type = VECTOR_TYPE; \
}; \
DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
///###

DECLARE_SIMD_AVX512_REGISTER(int8_t,    AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint8_t,   AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int16_t,   AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint16_t,  AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int32_t,   AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint32_t,  AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(int64_t,   AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(uint64_t,  AVX512, avx512_reg_i);
DECLARE_SIMD_AVX512_REGISTER(float,     AVX512, avx512_reg_f);
DECLARE_SIMD_AVX512_REGISTER(double,    AVX512, avx512_reg_d);

template <typename T, size_t W>
struct simd_avx512_bool_register {
    using scalar_t = bool;
    using arch_t = AVX512;
    using register_t = avx512_mask_traits_t<T>;
    // how many registers for this bool vector
    static constexpr size_t n_regs() {
        return W / 8 / sizeof(register_t);
    }
    static constexpr size_t reg_lanes() {
        return 8 * sizeof(register_t);
    }

    union alignas(n_regs() * sizeof(register_t)) {
        register_t regs_[n_regs()];
        std::bitset<W> bits_;
    };

    simd_avx512_bool_register() noexcept {}

    template <typename... Regs>
    simd_avx512_bool_register(register_t val, Regs... others) noexcept
        : regs_{val, others...}
    { }
    register_t reg(size_t idx = 0) const noexcept {
        return regs_[idx];
    }
    register_t& reg(size_t idx = 0) noexcept {
        return regs_[idx];
    }

    /// work with STL algorithm/iterator
    struct bitset_iterator {
        std::bitset<W>* bits_{nullptr};
        size_t npos_{-1};

        bitset_iterator(std::bitset<W>& bits)
            : bits_(&bits), npos_(bits.size())
        { }
        bitset_iterator(std::bitset<W>& bits, size_t npos)
            : bits_(&bits), npos_(npos)
        {
            assert(npos <= bits_.size());
        }

        using reference = typename std::bitset<W>::reference;
        reference operator *() {
            return (*bits_)[npos_];
        }

        bitset_iterator& operator++() noexcept {
            npos_++;
            return *this;
        }
        bool operator ==(const bitset_iterator& other) const noexcept {
            return bits_ == other.bits_ && npos_ == other.npos_;
        }
        bool operator !=(const bitset_iterator& other) const noexcept {
            return !(*this == other);
        }
    };

    struct const_bitset_iterator {
        std::bitset<W>* bits_{nullptr};
        size_t npos_{-1};

        const_bitset_iterator(std::bitset<W>& bits)
            : bits_(&bits), npos_(bits.size())
        { }
        const_bitset_iterator(std::bitset<W>& bits, size_t npos)
            : bits_(&bits), npos_(npos)
        {
            assert(npos <= bits_.size());
        }

        bool operator *() const {
            return (*bits_)[npos_];
        }

        const_bitset_iterator& operator++() noexcept {
            npos_++;
            return *this;
        }
        bool operator ==(const const_bitset_iterator& other) const noexcept {
            return bits_ == other.bits_ && npos_ == other.npos_;
        }
        bool operator !=(const const_bitset_iterator& other) const noexcept {
            return !(*this == other);
        }
    };

    using value_type = bool;
    using reference = typename std::bitset<W>::reference;
    using const_reference = const reference;
    using pointer = bitset_iterator;
    using const_pointer = const_bitset_iterator;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    scalar_t operator[](size_t idx) const noexcept {
        return bits_[idx];
    }
    reference& operator[](size_t idx) noexcept {
        return bits_[idx];
    }
    scalar_t at(size_t idx) const {
        return bits_[idx];
    }
    scalar_t get(size_t idx) const noexcept {
        return bits_[idx];
    }

    /// make vector iteratorable
    const_iterator begin() const noexcept {
        return bitset_iterator(bits_, 0);
    }
    const_iterator end() const noexcept {
        return bitset_iterator(bits_);
    }
    iterator begin() noexcept {
        return bitset_iterator(bits_, 0);
    }
    iterator end() noexcept {
        return bitset_iterator(bits_);
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }
    const_iterator cend() const noexcept {
        return end();
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }
    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }
    reverse_iterator rbegin() noexcept {
        return reverse_iterator(end());
    }
    reverse_iterator rend() noexcept {
        return reverse_iterator(begin());
    }

    const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(end());
    }
    const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(begin());
    }
};
template <typename T, size_t W>
struct get_bool_simd_register<T, W, AVX512>
{
    using type = simd_avx512_bool_register<T, W>;
};
}  // namespace types
}  // namespace simd
#endif  // SIMD_WITH_AVX512
