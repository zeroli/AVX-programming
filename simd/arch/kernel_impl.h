#pragma once

#include <cstddef>

namespace simd {
namespace kernel {
namespace impl {
#define DECLAE_OP_KERNEL(OP) \
template <typename T, size_t W, typename Enable = void> \
struct OP; \
///

DECLAE_OP_KERNEL(add);
DECLAE_OP_KERNEL(sub);
DECLAE_OP_KERNEL(mul);
DECLAE_OP_KERNEL(div);

DECLAE_OP_KERNEL(broadcast);

DECLAE_OP_KERNEL(bitwise_and);
DECLAE_OP_KERNEL(bitwise_or);
DECLAE_OP_KERNEL(bitwise_xor);
DECLAE_OP_KERNEL(bitwise_andnot);

DECLAE_OP_KERNEL(logical_and);
DECLAE_OP_KERNEL(logical_or);

DECLAE_OP_KERNEL(max);
DECLAE_OP_KERNEL(min);

DECLAE_OP_KERNEL(abs);
DECLAE_OP_KERNEL(sqrt);

#undef DECLAE_OP_KERNEL

}  // namespace impl
}  // namespace kernel
}  // namespace simd
