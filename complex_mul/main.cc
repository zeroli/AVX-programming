#include "io.h"

#include <iostream>

int main()
{
    auto x1 = _mm256_setr_pd(4.0, 5.0, 6.0, 7.0);
    auto x2 = _mm256_setr_pd(9.0, 3.0, 6.0, 7.0);
    auto neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    std::cout << "x1: " << x1 << "\n";
    std::cout << "x2: " << x2 << "\n";

    /*
    cp1_2 : a b x y
    cp3_4:  c d z w
    (a + bi) * (c + di) = ac - bd + (ad + bc)i
    (x + yi) * (z + wi) = xz - yw + (xw + yz)i

    cp1_2 * cp3_4:
    ac-bd ad+bc xz-yw xw+yz

    直接相乘：
        ac bd xz yw
    要等到最后结果：
        ac-bd ad+bc xz-yw xw+yz
    采用水平相减：
    => hsub_pd: vec1, vec2 ?
        ac-bd (?) xz-yw (?)
        ac-bd (ad-(-bc)) xz-yw (xw-(-yz))
    => 构造vec2
        vec2: ad -bc xw -yz
           a  b  x  y  (原输入1)
    (*)   d -c  w -z
           c  d  z  w    (原输入2)
           swap c/d, z/w, 采用permute
    =>  d  c  w  z
    (*)   1 -1 1  -1
           d -c -w -z
    */
    auto m1 = _mm256_mul_pd(x1, x2);
    std::cout << "x1*x2: " << m1 << "\n";

    x2 = _mm256_permute_pd(x2, 0b0101);
    std::cout << "perm(x2): " << x2 << "\n";

    x2 = _mm256_mul_pd(x2, neg);
    std::cout << "neg(perm(x2)): " << x2 << "\n";

    x2 = _mm256_mul_pd(x1, x2);
    std::cout << "x1 * neg(perm(x2)): " << x2 << "\n";

   auto ret = _mm256_hsub_pd(m1, x2);
   std::cout << "hsub(x1*x2, x1 * neg(perm(x2))) = " << ret << "\n";

   std::cout << "(4 + 5i) * (9 + 3i) = " << (36 - 15) << "+" << (12 + 45) << "i" << "\n";
   std::cout << "(6 + 7i) * (6 + 7i) = " << (36 - 49) << "+" << (42 + 42) << "i" << "\n";
}
