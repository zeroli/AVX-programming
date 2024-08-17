#include "immintrin.h"

#include "io.h"

#include <iostream>
#include <vector>
#include <string>

void permute_f32()
{
    std::cout << "=> _mm256_permute_ps" << "\n";

    __m256 x = _mm256_set_ps(7.f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.f);
    std::cout << "x = " << simd::Make(x) << "\n";
    __m256 y = _mm256_permute_ps(x, 0b01110100);
    std::cout << "permute at " << "00 01 11 01 00 01 11 01" << "\n";
    std::cout << "y = " << simd::Make(y) << "\n\n";
    /*
    x = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=4, [5]=5, [6]=6, [7]=7}
    permute at 00 01 11 01 00 01 11 01
    y = vecf32x8{[0]=0, [1]=1, [2]=3, [3]=1, [4]=4, [5]=5, [6]=7, [7]=5}
    */
}

void permute_f64()
{
    std::cout << "=> _mm256_permute_pd" << "\n";

    __m256d x = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    std::cout << "x = " << simd::Make(x) << "\n";
    __m256d y = _mm256_permute_pd(x, 0b00000100);
    // 这里低4个bits来选择，前2bits从前2个数中选择，后2bits从后2个数中选择
    // 每2个bits从一对相邻值选择
    // bit0: 0, 选择前2个数中第一个数 => 0.0
    // bit1: 0, 选择前2个数中第一个数 => 0.0
    // bit2: 1, 选择后2个数中第二个数 => 3.0
    // bit3: 0, 选择后2个数中第一个数 => 2.0
    std::cout << "permute at " << "00 01 xx xx" << "\n";
    std::cout << "y = " << simd::Make(y) << "\n\n";
    /*
    x = vecf64x4{[0]=0, [1]=1, [2]=2, [3]=3}
    permuate at 00 01 xx xx
    y = vecf64x4{[0]=0, [1]=0, [2]=3, [3]=2}
    */
}

void permute4x64_f64()
{
    std::cout << "=> _mm256_permute4x64_pd" << "\n";
    __m256d x = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    std::cout << "x = " << simd::Make(x) << "\n";
    __m256d y = _mm256_permute4x64_pd(x, 0b10010100);
    // 这里是每2bits(共4个选择)来选择输入4个值其中一个
    std::cout << "permute at " << "00(0) 01(1) 01(1) 10(2)" << "\n";
    std::cout << "y = " << simd::Make(y) << "\n\n";
}

void permute2f128_f32()
{
    std::cout << "=> _mm256_permute2f128_ps" << "\n";
    auto x1 = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    auto x2 = _mm256_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.f, 9.0f, 8.0f);
    std::cout << "x1 = " << simd::Make(x1) << "\n";
    std::cout << "x2 = " << simd::Make(x2) << "\n";
    {
        auto y = _mm256_permute2f128_ps(x1, x2, 0b00110100);
        // 这里是每4bits一起看: select4(x1, x2, 4bits)
        // 前4bits中，低2bits用来选择4个128bits中的其中一个
        // 如果第3bits为1，则选择结果置为0
        // 后4bits中，低2bits用来选择4个128bits中的其中一个
        // 如果第3bits为1，则选择结果置为0
        std::cout << "permute at " << "00(0) 01(not zero) 11(3) 00(not zero)" << "\n";
        std::cout << "y = " << simd::Make(y) << "\n\n";
        /*
        x1 = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=4, [5]=5, [6]=6, [7]=7}
        x2 = vecf32x8{[0]=8, [1]=9, [2]=10, [3]=11, [4]=12, [5]=13, [6]=14, [7]=15}
        permute at 00(0) 01(not zero) 11(3) 00(not zero)
        y = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=12, [5]=13, [6]=14, [7]=15}
        */
    }
    {
        auto y = _mm256_permute2f128_ps(x1, x2, 0b10110100);
        // 第二次选择中，因为第3bit为1，选择结果置为0
        std::cout << "permute at " << "00(0) 01(not zero) 11(3) 00(zero)" << "\n";
        std::cout << "y = " << simd::Make(y) << "\n\n";
        /*
        x1 = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=4, [5]=5, [6]=6, [7]=7}
        x2 = vecf32x8{[0]=8, [1]=9, [2]=10, [3]=11, [4]=12, [5]=13, [6]=14, [7]=15}
        permute at 00(0) 01(not zero) 11(3) 00(zero)
        y = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=0, [5]=0, [6]=0, [7]=0}
        */
    }
}

void permutevar_f32()
{
    std::cout << "=> _mm256_permutevar_ps" << "\n";
    auto x = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    auto c = _mm256_set_epi32(0b11, 0b10, 0b01, 0b00, 0b00, 0b01, 0b10, 0b11);
    std::cout << "x1 = " << simd::Make(x) << "\n";
    {
        auto y = _mm256_permutevar_ps(x, c);
        // 目标结果256位，8个f32的数
        // 每个数(32bits, 4 bytes)，从输入中的前128bits和后128bits来选
        // 前4个数，从前128bits来选，后4个数，从后128bits来选
        // 每个数，从128bits中来选，有4个数供选择，只需要2bits(00, 01, 10, 11)
        // 这里，前4个数倒序来选择，后4个数正序来选择
        // 输入选择也是一个256bits的数，共8个整数，每个整数只需要贡献低2bits就可以了
        std::cout << "permute at " << "11(3) 10(2) 01(1) 00(0) 00(0) 01(1) 10(2) 11(3)" << "\n";
        std::cout << "y = " << simd::Make(y) << "\n\n";
        /*
        x1 = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=4, [5]=5, [6]=6, [7]=7}
        permute at 11(3) 10(2) 01(1) 00(0) 00(0) 01(1) 10(2) 11(3)
        y = vecf32x8{[0]=3, [1]=2, [2]=1, [3]=0, [4]=4, [5]=5, [6]=6, [7]=7}
        */
    }
}

void permutevar8x32_f32()
{
    std::cout << "=> _mm256_permutevar8x32_ps" << "\n";
    auto x = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    auto c = _mm256_set_epi32(0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111);
    std::cout << "x1 = " << simd::Make(x) << "\n";
    {
        auto y = _mm256_permutevar8x32_ps(x, c);
        // 目标结果256位，8个f32的数
        // 每个数从输入8个数中选择，8个选择只需要3bits
        // 因此输入控制数每个数只需要看低3bits就够了
        // 这里是倒序选择每个数
        std::cout << "permute at " << "111(7) 110(6) 101(5) 100(4) 011(3) 010(2) 001(1) 000(0)" << "\n";
        std::cout << "y = " << simd::Make(y) << "\n\n";
        /*
        x1 = vecf32x8{[0]=0, [1]=1, [2]=2, [3]=3, [4]=4, [5]=5, [6]=6, [7]=7}
        permute at 111(7) 110(6) 101(5) 100(4) 011(3) 010(2) 001(1) 000(0)
        y = vecf32x8{[0]=7, [1]=6, [2]=5, [3]=4, [4]=3, [5]=2, [6]=1, [7]=0}
        */
    }
}

int main()
{
    permute_f32();
    permute_f64();
    permute4x64_f64();
    permute2f128_f32();
    permutevar_f32();
    permutevar8x32_f32();
}
