#include <gtest/gtest.h>

#include "simd/memory/aligned_allocator.h"

TEST(aligned_allocator, test_1)
{
    simd::aligned_allocator<int, 32> alloc;
    auto ptr = alloc.allocate(10);
    EXPECT_TRUE(ptr != nullptr);
    EXPECT_TRUE(simd::is_aligned(ptr, 32));
}
