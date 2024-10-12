#define DECLARE_OP_KERNEL(OP) \
template <typename T, size_t W, typename Enable = void> \
struct OP; \
///

/// arithmetic op kernels
DECLARE_OP_KERNEL(add);
DECLARE_OP_KERNEL(sub);
DECLARE_OP_KERNEL(mul);
DECLARE_OP_KERNEL(div);
DECLARE_OP_KERNEL(mod);
DECLARE_OP_KERNEL(neg);

/// FMA kernels
DECLARE_OP_KERNEL(fmadd);
DECLARE_OP_KERNEL(fmsub);
DECLARE_OP_KERNEL(fnmadd);
DECLARE_OP_KERNEL(fnmsub);
DECLARE_OP_KERNEL(fmaddsub);
DECLARE_OP_KERNEL(fmsubadd);

/// memory IO kernels
DECLARE_OP_KERNEL(set);
DECLARE_OP_KERNEL(setzero);
DECLARE_OP_KERNEL(to_mask);
DECLARE_OP_KERNEL(from_mask);
DECLARE_OP_KERNEL(load_aligned);
DECLARE_OP_KERNEL(load_unaligned);
DECLARE_OP_KERNEL(store_aligned);
DECLARE_OP_KERNEL(store_unaligned);
DECLARE_OP_KERNEL(broadcast);
DECLARE_OP_KERNEL(load_complex);
DECLARE_OP_KERNEL(complex_packlo);
DECLARE_OP_KERNEL(complex_packhi);

/// bitwise op kernels
DECLARE_OP_KERNEL(bitwise_and);
DECLARE_OP_KERNEL(bitwise_or);
DECLARE_OP_KERNEL(bitwise_xor);
DECLARE_OP_KERNEL(bitwise_andnot);
DECLARE_OP_KERNEL(bitwise_not);
DECLARE_OP_KERNEL(bitwise_lshift);
DECLARE_OP_KERNEL(bitwise_rshift);

/// comparison op kernels
DECLARE_OP_KERNEL(eq);
DECLARE_OP_KERNEL(ne);
DECLARE_OP_KERNEL(gt);
DECLARE_OP_KERNEL(ge);
DECLARE_OP_KERNEL(lt);
DECLARE_OP_KERNEL(le);

/// math function kernels
DECLARE_OP_KERNEL(ceil);
DECLARE_OP_KERNEL(floor);
DECLARE_OP_KERNEL(abs);
DECLARE_OP_KERNEL(sqrt);
DECLARE_OP_KERNEL(exp);
DECLARE_OP_KERNEL(exp10);
DECLARE_OP_KERNEL(exp2);
DECLARE_OP_KERNEL(expm1);
DECLARE_OP_KERNEL(hypot);
DECLARE_OP_KERNEL(log);
DECLARE_OP_KERNEL(log2);
DECLARE_OP_KERNEL(log10);
DECLARE_OP_KERNEL(log1p);
DECLARE_OP_KERNEL(avgr);
DECLARE_OP_KERNEL(avg);

/// trigo function kernels
DECLARE_OP_KERNEL(sin);
DECLARE_OP_KERNEL(cos);
DECLARE_OP_KERNEL(sincos);
DECLARE_OP_KERNEL(tan);
DECLARE_OP_KERNEL(asin);
DECLARE_OP_KERNEL(acos);
DECLARE_OP_KERNEL(atan);
DECLARE_OP_KERNEL(atan2);

/// hyperbolic function kernels
DECLARE_OP_KERNEL(sinh);
DECLARE_OP_KERNEL(cosh);
DECLARE_OP_KERNEL(tanh);
DECLARE_OP_KERNEL(asinh);
DECLARE_OP_KERNEL(atanh);

/// algorithm kernels
DECLARE_OP_KERNEL(max);
DECLARE_OP_KERNEL(min);
DECLARE_OP_KERNEL(sign);
DECLARE_OP_KERNEL(bitofsign);
DECLARE_OP_KERNEL(copysign);
DECLARE_OP_KERNEL(all_of);
DECLARE_OP_KERNEL(any_of);
DECLARE_OP_KERNEL(none_of);
DECLARE_OP_KERNEL(some_of);
DECLARE_OP_KERNEL(popcount);
DECLARE_OP_KERNEL(find_first_set);
DECLARE_OP_KERNEL(find_last_set);
DECLARE_OP_KERNEL(select);

DECLARE_OP_KERNEL(hadd);
DECLARE_OP_KERNEL(reduce_sum);
DECLARE_OP_KERNEL(reduce_max);
DECLARE_OP_KERNEL(reduce_min);

template <typename T, size_t W, typename F, typename Enable = void>
struct reduce;

template <typename U, typename T, size_t W, typename Enable = void>
struct cast;

template <typename T, size_t W, typename U, typename V, typename Enable = void>
struct gather;
template <typename T, size_t W, typename U, typename V, typename Enable = void>
struct scatter;

#undef DECLARE_OP_KERNEL
