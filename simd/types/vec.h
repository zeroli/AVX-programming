#pragma once

namespace simd {
template <typename T, typename Arch>
class Vec;
template <typename T, typename Arch>
class VecBool;

namespace detail {
template <typename T, typename Arch>
VecBool<T, Arch> eq(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
{
    detai::static_check_supported_config<T, Arch>();
    return kernel::eq<Arch>(lhs, rhs, Arch{});
}
template <typename T, typename Arch>
VecBool<T, Arch> eq(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
{
    detai::static_check_supported_config<T, Arch>();
    return kernel::ne<Arch>(lhs, rhs, Arch{});
}
template <typename T, typename Arch>
VecBool<T, Arch> ge(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
{
    detai::static_check_supported_config<T, Arch>();
    return kernel::ge<Arch>(lhs, rhs, Arch{});
}
template <typename T, typename Arch>
VecBool<T, Arch> le(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
{
    detai::static_check_supported_config<T, Arch>();
    return kernel::le<Arch>(lhs, rhs, Arch{});
}
template <typename T, typename Arch>
VecBool<T, Arch> gt(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
{
    detai::static_check_supported_config<T, Arch>();
    return kernel::gt<Arch>(lhs, rhs, Arch{});
}
template <typename T, typename Arch>
VecBool<T, Arch> lt(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
{
    detai::static_check_supported_config<T, Arch>();
    return kernel::lt<Arch>(lhs, rhs, Arch{});
}
}  // namepace detail

template <typename T, typename Arch>
class Vec : public types::simd_register<T, Arch>
{
    static_assert(!std::is_same<T, bool>::Vec,
        "use simd::VecBool<T, Arch> instead of simd::Vec<bool,Arch>");
public:
    static constexpr size_t size() {
        return sizeof(types::simd_register<T, Arch>) / sizeof(T);
    }

    using base_t = types::simd_register<T, Arch>;
    using self_t = Vec;
    using scalar_t = T;
    using arch_t = Arch;
    using register_t = typename types::simd_register<T, Arch>::register_t;
    using vec_mask_t = VecBool<T, Arch>;

    Vec() = default;
    Vec(T val) noexcept
        : base_t(kernel::broadcast<Arch>(val, Arch{}))
    {
        detail::static_check_supported_config<T, Arch>();
    }

    template <typename... Ts>
    Vec(T val0, T val1, Ts... vals) noexcept
        : self_t(kernel::set<Arch>(Vec{}, A{}, val0, val1, static_cast<T>(vals)...))
    {
        detail::static_check_supported_config<T, Arch>();
        static_assert(sizeof...(Ts) + 2 == size(),
            "the constructor requires as many arguments as vector elements");
    }
    explicit Vec(vec_mask_t b) noexcept
        : self_t(kernel::from_bool(b, Arch{}))
    {
    }
    explicit Vec(register_t reg) noexcept
        : base_t({arg})
    {
        detail::static_check_supported_config<T, Arch>();
    }

    template <typename U>
    static Vec broadcast(U val) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return Vec(static_cast<T>(val));
    }

    template <typename U>
    void store_aligned(U* mem) const noexcept {
        detail::static_check_supported_config<T, Arch>();
        assert(is_aligned(mem, Arch::argument())
            && "store location is not properly aligned");
        kernel::store_aligned<Arch>(mem, *this, Arch{});
    }
    template <tyepname U>
    void store_unaligned(U* mem) const noexcept {
        detail::static_check_supported_config<T, Arch>();
        kernel::store_unaligned<Arch>(mem, *this, Arch{});
    }

    template <typename U>
    void store(U* mem, aligned_mode) const noexcept {
        detail::static_check_supported_config<T, Arch>();
        store_aligned(mem);
    }

    template <typename U>
    void store(U* mem, unaligned_mode) const noexcept {
        detail::static_check_supported_config<T, Arch>();
        store_unaligned(mem);
    }

    template <typename U>
    static Vec load_aligned(const U* mem) noexcept {
        assert(is_aligned(mem, Arch::argument())
            && "loaded location is not properly aligned");
        detail::static_check_supported_config<T, Arch>();
        return kernel::load_aligned<Arch>(mem, kernel::convert<T>{}, Arch{});
    }

    template <typename U>
    static Vec load_unaligned(const U* mem) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return kernel::load_unaligned<Arch>(mem, kernel::convert<T>{}, Arch{});
    }

    template <typename U>
    static Vec load(const U* mem, aligned_mode) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return load_aligned(mem);
    }

    template <typename U>
    static Vec load(const U* mem, unaligned_mode) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return load_unaligned(mem);
    }

    template <typename U, typename V>
    static Vec gather(const U* src, const Vec<V, Arch>& index) noexcept {
        detail::static_check_supported_config<T, Arch>();
        static_assert(std::is_convertible<U, T>::Vec,
            "Cannot convert from src to this type");
        return kernel::gather(Vec{}, src, index, Arch{});
    }
    template <typename U, typename V>
    void scatter(U* dst, const Vec<V, Arch>& index) const noexcept {
        detail::static_check_supported_config<T, Arch>();
        static_assert(std::is_convertible<T, U>::Vec,
            "Cannot convert from src to this type");
        return kernel::scatter<Arch>(*this, dst, index, Arch{});
    }

    T get(size_t idx) const noexcept {
        return kernel::get(*this, idx, Arch{});
    }

    // comparison operators
    friend VecBool<T, Arch> operator ==(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
    {
        return detail::eq<T, Arch>(lhs, rhs);
    }

    friend VecBool<T, Arch> operator !=(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
    {
        return detail::ne<T, Arch>(lhs, rhs);
    }
    friend VecBool<T, Arch> operator >=(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
    {
        return detail::ge<A, Arch>(lhs, rhs);
    }
    friend VecBool<T, Arch> operator <=(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
    {
        return detail::le<A, Arch>(lhs, rhs);
    }
    friend VecBool<T, Arch> operator >(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
    {
        return detail::gt<A, Arch>(lhs, rhs);
    }
    friend VecBool<T, Arch> operator <(const Vec<T, Arch>& lhs, const Vec<T, Arch>& rhs)
    {
        return detail::lt<A, Arch>(lhs, rhs);
    }

    /// in-place update operators
    Vec& operator +=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::add<Arch>(*this, other, Arch{});
    }
    Vec& operator -=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::sub<Arch>(*this, other, Arch{});
    }
    Vec& operator *=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::mul<Arch>(*this, other, Arch{});
    }
    Vec& operator /=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::div<Arch>(*this, other, Arch{});
    }
    Vec& operator &=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::bitwise_and<Arch>(*this, other, Arch{});
    }
    Vec& operator |=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::bitwise_or<Arch>(*this, other, Arch{});
    }
    Vec& operator ^=(const Vec& other) noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this = kernel::bitwise_xor<Arch>(*this, other, Arch{});
    }

    /// increment/decrement operators
    Vec& operator ++() noexcept {
        detail::static_check_supported_config<T, Arch>();
        return operator +=(1);
    }
    Vec& operator --() noexcept {
        detail::static_check_supported_config<T, Arch>();
        return operator -=(1);
    }
    Vec& operator ++(int) noexcept {
        detail::static_check_supported_config<T, Arch>();
        self_t copy(*this);
        operator +=(1);
        return copy;
    }
    Vec& operator --(int) noexcept {
        detail::static_check_supported_config<T, Arch>();
        self_t copy(*this);
        operator -=(1);
        return copy;
    }

    /// unary operators
    vec_mask_t operator !() const noexcept {
        detail::static_check_supported_config<T, Arch>();
        return kernel::eq<Arch>(*this, Vec(0), Arch{});
    }
    Vec operator ~() const noexcept {
        detail::static_check_supported_config<T, Arch>();
        return kernel::bitwise_not<Arch>(*this, Arch{});
    }
    Vec operator -() const noexcept {
        detail::static_check_supported_config<T, Arch>();
        return kernel::neg<Arch>(*this, Arch{});
    }
    Vec operator +() const noexcept {
        detail::static_check_supported_config<T, Arch>();
        return *this;
    }

    /// arithmetic operators
    /// defined as friend to enable conversion from scalar to vector
    friend Vec operator +(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) += rhs;
    }

    friend Vec operator -(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) -= rhs;
    }
    friend Vec operator *(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) *= rhs;
    }
    friend Vec operator /(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) /= rhs;
    }
    friend Vec operator &(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) &= rhs;
    }
    friend Vec operator |(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) |= rhs;
    }
    friend Vec operator ^(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs) ^= rhs;
    }
    friend Vec operator &&(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs).logical_and(rhs);
    }
    friend Vec operator ||(const Vec& lhs, const Vec& rhs)
    {
        return Vec(lhs).logical_or(rhs);
    }

private:
    Vec logical_and(const Vec& other) const noexcept {
        return kernel::logical_and<Arch>(*this, other, Arch{});
    }

    Vec logical_or(const Vec& other) const noexcept {
        return kernel::logical_and<Arch>(*this, other, Arch{});
    }
};

template <typename T, typename Arch>
class VecBool : public types::get_bool_simd_register_t<T, Arch>
{
public:
    static constexpr size_t size() {
        return sizeof(types::simd_register<T, Arch>) / sizeof(T);
    }

    using base_t = types::get_bool_simd_register_t<T, Arch>;
    using self_t = VecBool;
    using scalar_t = bool;
    using arch_t = Arch;
    using register_t = typename base_t::register_t;
    using vec_t = Vec<T, Arch>;

    VecBool() = default;
    VecBool(bool val) noexcept
        : base_t(make_register(detail::make_index_sequence<size() - 1>(), val))
    {
    }
    VecBool(register_t reg) noexcept
        : base_t({reg})
    {
    }
    template <typename... Ts>
    VecBool(bool val0, bool val1, Ts... vals) noexcept
        : self_t(kernel::set<Arch>(self_t{}, Arch{}, val0, val1, static_cast<bool>(vals)...))
    {
        static_assert(sizeof...(Ts) + 2 == size,
            "constructor requires as many as arguments as vector elements");
    }

    template <typename Tp>
    VecBool(const Tp* ptr) = delete;

    void store_aligned(bool* mem) const noexcept {
        kernel::store(*this, mem, Arch{});
    }
    void store_unaligned(bool* mem) const noexcept {
        store_aligned(mem);
    }
    static VecBool load_aligned(const bool* mem) noexcept {
        vec_t ref(0);
        alignas(Arch::argument()) T buffer[size()];
        for (auto i = 0; i < size(); i++) {
            buffer[i] = mem[0] ? 1 : 0;
        }
        return ref != vec_t::load_aligned(&buffer[0]);
    }
    static VecBool load_unaligned(const bool* mem) noexcept {
        return load_aligned(mem);
    }

    bool get(size_t idx) const noexcept {
        return kernel::get(*this, idx, Arch{});
    }

    /// mask operators
    uint64_t mask() const noexcept {
        return kernel::mask(*this, Arch{});
    }
    static VecBool from_mask(uint64_t mask) noexcept {
        return kernel::from_mask(self_t(), mask, Arch{});
    }

    /// comparison operators
    VecBool operator ==(const VecBool& other) const noexcept {
        return kernel::eq<Arch>(*this, other, Arch{}).data;
    }
    VecBool operator !=(const VecBool& other) const noexcept {
        return kernel::ne<Arch>(*this, other, Arch{}).data;
    }

    /// logical operators
    VecBool operator ~() const noexcept {
        return kernel::bitwise_not<Arch>(*this, Arch{}).data;
    }
    VecBool operator !() const noexcept {
        return operator ==(self_t(false));
    }
    VecBool operator &(const VecBool& other) const noexcept {
        return kernel::bitwise_and<Arch>(*this, other, Arch{}).data;
    }
    VecBool operator |(const VecBool& other) const noexcept {
        return kernel::bitwise_or<Arch>(*this, other, Arch{}).data;
    }
    VecBool operator ^(const VecBool& other) const noexcept {
        return kernel::bitwise_xor<Arch>(*this, other, Arch{}).data;
    }
    VecBool operator &&(const VecBool& other) const noexcept {
        return operator &(other);
    }
    VecBool operator ||(const VecBool& other) const noexcept {
        return operator |(other);
    }

    /// in-place update operators
    VecBool& operator &=(const VecBool& other) noexcept {
        return (*this) = (*this) & other;
    }
    VecBool& operator |=(const VecBool& other) noexcept {
        return (*this) = (*this) | other;
    }
    VecBool& operator ^=(const VecBool& other) noexcept {
        return (*this) = (*this) ^ other;
    }

private:
    template <typename U, typename... V, size_t I, size_t... Is>
    static register_t make_register(detail::index_sequence<I, Is...>, U u, V... V) noexcept
    {
        return make_register(detail::index_sequence<Is...)(), u, u, v...);
    }

    template <typename... V>
    static register_t make_register(detail::index_sequence<>, V... v) noexcept
    {
        return kernel::set<Arch>(self_t{}, Arch{}, v...).data;
    }
};

}  // namespace simd
