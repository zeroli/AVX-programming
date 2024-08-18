#pragma once

#include "simd/memory/alignment.h"

#include <cstddef>
#include <cassert>
#include <memory>

namespace simd {
inline void* aligned_malloc(size_t alignment, size_t size)
{
    assert((alignment & (alignment - 1)) == 0
        && "alignment must be power of 2");
    assert((alignment >= sizeof(void*))
        && "alignment must be greater than sizeof(void*)");
    void* res = nullptr;
#ifdef _WIN32
    res = _aligned_malloc(size, alignment);
#else
    posix_memalign(&res, alignment, size);
#endif
    return res;
}

inline void aligned_free(void* ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

template <typename T, size_t Alignment>
class aligned_allocator
{
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static constexpr size_t alignment = Alignment;

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;
    aligned_allocator(const aligned_allocator& rhs) noexcept = default;

    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>& rhs) noexcept
    { }

    ~aligned_allocator() = default;

    pointer address(reference r) noexcept {
        return &r;
    }
    const_pointer address(const_reference r) noexcept {
        return &r;
    }

    pointer allocate(size_type n, const void* hint = 0) {
        pointer res = reinterpret_cast<pointer>(
            aligned_malloc(Alignment, sizeof(T) * n));
        if (res == nullptr) {
            throw std::bad_alloc();
        }
        return res;
    }
    void deallocate(pointer p, size_type n) {
        aligned_free(p);
    }

    size_type max_size() const noexcept {
        return size_type(-1) / sizeof(T);
    }
    size_type size_max() const noexcept {
        return size_type(-1) / sizeof(T);
    }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new ((void*)p) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p) {
        p->~U();
    }
};

template <typename T1, size_t Alignment1, typename T2, size_t Alignment2>
inline bool operator ==(const aligned_allocator<T1, Alignment1>& lhs,
                        const aligned_allocator<T2, Alignment2>& rhs)
{
    return lhs.alignment == rhs.alignment;
}

template <typename T1, size_t Alignment1, typename T2, size_t Alignment2>
inline bool operator !=(const aligned_allocator<T1, Alignment1>& lhs,
                        const aligned_allocator<T2, Alignment2>& rhs)
{
    return !(lhs == rhs);
}

}  // namespace simd
