#pragma once

#ifdef __AVX__
#define SIMD_WITH_AVX 1
#else
#define SIMD_WITH_AVX 0
#endif

#ifdef __AVX2__
#define SIMD_WITH_AVX2 1
#else
#define SIMD_WITH_AVX2 0
#endif

#ifdef __FMA__

#ifdef __AVX__
#define SIMD_WITH_FMA3_AVX 1
#else
#define SIMD_WITH_FMA3_AVX 0
#endif  // __AVX__

#ifdef __AVX2__
#define SIMD_WITH_FMA3_AVX2 1
#else
#define SIMD_WITH_FMA3_AVX2 0
#endif  // __AVX__

#endif  // __FMA__

#ifdef __FMA4__
#define SIMD_WITH_FMA4  1
#else
#define SIMD_WITH_FMA4  0
#endif  // __FMA4__

#ifdef __AVX512F__
#define SIMD_WITH_AVX512F 1
#else
#define SIMD_WITH_AVX512F 0
#endif  // __AVX512F__

#ifdef __AVX512CD__
#define SIMD_WITH_AVX512CD SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512CD 0
#endif  // __AVX512CD__

#ifdef __AVX512DQ__
#define SIMD_WITH_AVX512DQ SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512DQ 0
#endif  // __AVX512DQ__

#ifdef __AVX512BW__
#define SIMD_WITH_AVX512BW SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512BW 0
#endif  // __AVX512BW__

#ifdef __AVX512ER__
#define SIMD_WITH_AVX512ER SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512ER 0
#endif  // __AVX512ER__

#ifdef __AVX512PF__
#define SIMD_WITH_AVX512PF SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512PF 0
#endif  // __AVX512PF__

#ifdef __AVX512IFMA__
#define SIMD_WITH_AVX512IFMA SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512IFMA 0
#endif  // __AVX512IFMA__

#ifdef __AVX512VBMI__
#define SIMD_WITH_AVX512VBMI SIMD_WITH_AVX512F
#else
#define SIMD_WITH_AVX512VBMI 0
#endif  // __AVX512VBMI__
