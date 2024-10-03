#pragma once

namespace simd {
namespace ut {

/// use macro instead of template for this function
/// gtest will report failed test with function file/line
/// which is not easy to identify which call trigger failed case
#define TEST_VEC_TYPE(VT, CHECK_SIZE, CHECK_N_REGS, CHECK_REG_LANES, CHECK_ARCH) \
{ \
    using vec_t = VT; \
    EXPECT_EQ(true, (std::is_same<CHECK_ARCH, typename vec_t::arch_t>::value)); \
    EXPECT_EQ(CHECK_SIZE, vec_t::size()); \
    EXPECT_EQ(CHECK_N_REGS, vec_t::n_regs()); \
    EXPECT_EQ(CHECK_REG_LANES, vec_t::reg_lanes()); \
    EXPECT_EQ(vec_t::size(), vec_t::n_regs() * vec_t::reg_lanes()); \
} \
///###

}  // namespace ut
}  // namespace simd
