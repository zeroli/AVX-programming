#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wignored-attributes"
#elif defined(__clang__)
#pragma clang diagnostic ignored "-Wignored-attributes"
#endif

/// minimum supported ISA: SSE for all SSE (>= SSE4.x)
#define SIMD_WITH_SSE 1

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

/// -mfma set, __FMA__ defined, and __AVX__ defined as well
/// and all avx intrinsics are available to use
#ifdef __FMA__
#define SIMD_WITH_FMA3_SSE SIMD_WITH_SSE
#define SIMD_WITH_FMA3_AVX SIMD_WITH_AVX
#define SIMD_WITH_FMA3_AVX2 SIMD_WITH_AVX2
#else
#define SIMD_WITH_FMA3_SSE 0
#define SIMD_WITH_FMA3_AVX 0
#define SIMD_WITH_FMA3_AVX2 0
#endif  // __FMA__

/// compiler flags below enable these macro definitions
/// -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl
/// we bunch them to form one flag to enable avx512
#if defined(__AVX512F__) && defined(__AVX512CD__) && \
    defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define SIMD_WITH_AVX512 1
#else
#define SIMD_WITH_AVX512 0
#endif  // __AVX512__
