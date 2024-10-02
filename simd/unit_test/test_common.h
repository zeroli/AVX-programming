#pragma once

namespace simd {
namespace ut {

template <typename VT,
    size_t CHECK_SIZE,
    size_t CHECK_N_REGS,
    size_t CHECK_REG_LANES,
    typename CHECK_ARCH>
static void test_type()
{
    using vec_t = VT;
    EXPECT_EQ(true, (std::is_same<CHECK_ARCH, typename vec_t::arch_t>::value));
    EXPECT_EQ(CHECK_SIZE, vec_t::size());
    EXPECT_EQ(CHECK_N_REGS, vec_t::n_regs());
    EXPECT_EQ(CHECK_REG_LANES, vec_t::reg_lanes());
    EXPECT_EQ(vec_t::size(), vec_t::n_regs() * vec_t::reg_lanes());
}
}  // namespace ut
}  // namespace simd
