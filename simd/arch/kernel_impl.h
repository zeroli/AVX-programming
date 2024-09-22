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

/// memory IO kernels
DECLARE_OP_KERNEL(set);
DECLARE_OP_KERNEL(to_mask);
DECLARE_OP_KERNEL(from_mask);
DECLARE_OP_KERNEL(load_aligned);
DECLARE_OP_KERNEL(load_unaligned);
DECLARE_OP_KERNEL(store_aligned);
DECLARE_OP_KERNEL(store_unaligned);
DECLARE_OP_KERNEL(broadcast);

/// bitwise op kernels
DECLARE_OP_KERNEL(bitwise_and);
DECLARE_OP_KERNEL(bitwise_or);
DECLARE_OP_KERNEL(bitwise_xor);
DECLARE_OP_KERNEL(bitwise_andnot);
DECLARE_OP_KERNEL(bitwise_not);
DECLARE_OP_KERNEL(bitwise_lshift);
DECLARE_OP_KERNEL(bitwise_rshift);

DECLARE_OP_KERNEL(logical_and);
DECLARE_OP_KERNEL(logical_or);

/// comparison op kernels
DECLARE_OP_KERNEL(eq);
DECLARE_OP_KERNEL(ne);
DECLARE_OP_KERNEL(gt);
DECLARE_OP_KERNEL(ge);
DECLARE_OP_KERNEL(lt);
DECLARE_OP_KERNEL(le);

/// math function kernels
DECLARE_OP_KERNEL(abs);
DECLARE_OP_KERNEL(sqrt);
DECLARE_OP_KERNEL(ceil);
DECLARE_OP_KERNEL(floor);
DECLARE_OP_KERNEL(sign);
DECLARE_OP_KERNEL(bitofsign);
DECLARE_OP_KERNEL(copysign);

/// algorithm kernels
DECLARE_OP_KERNEL(max);
DECLARE_OP_KERNEL(min);
DECLARE_OP_KERNEL(all);
DECLARE_OP_KERNEL(any);
DECLARE_OP_KERNEL(select);

template <typename U, typename T, size_t W, typename Enable = void>
struct cast;

#undef DECLARE_OP_KERNEL
