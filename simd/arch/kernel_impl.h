#define DECLAE_OP_KERNEL(OP) \
template <typename T, size_t W, typename Enable = void> \
struct OP; \
///

/// arithmetic op kernels
DECLAE_OP_KERNEL(add);
DECLAE_OP_KERNEL(sub);
DECLAE_OP_KERNEL(mul);
DECLAE_OP_KERNEL(div);
DECLAE_OP_KERNEL(neg);

DECLAE_OP_KERNEL(broadcast);

/// bitwise op kernels
DECLAE_OP_KERNEL(bitwise_and);
DECLAE_OP_KERNEL(bitwise_or);
DECLAE_OP_KERNEL(bitwise_xor);
DECLAE_OP_KERNEL(bitwise_andnot);
DECLAE_OP_KERNEL(bitwise_not);

DECLAE_OP_KERNEL(logical_and);
DECLAE_OP_KERNEL(logical_or);

/// comparison op kernels
DECLAE_OP_KERNEL(eq);
DECLAE_OP_KERNEL(ne);
DECLAE_OP_KERNEL(gt);
DECLAE_OP_KERNEL(ge);
DECLAE_OP_KERNEL(lt);
DECLAE_OP_KERNEL(le);

/// math function kernels
DECLAE_OP_KERNEL(abs);
DECLAE_OP_KERNEL(sqrt);

/// algorithm kernels
DECLAE_OP_KERNEL(max);
DECLAE_OP_KERNEL(min);
DECLAE_OP_KERNEL(all);
DECLAE_OP_KERNEL(any);

#undef DECLAE_OP_KERNEL
