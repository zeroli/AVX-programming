#include <gtest/gtest.h>

#include "simd/util/util.h"

int main(int argc, char** argv)
{
    simd::util::DumpArchRegConfig(std::cerr);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
