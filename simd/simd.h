#pragma once

#ifdef __cpp_if_constexpr
#define SIMD_IF_CONSTREXPR if constexpr
#endif

#if !defined(SIMD_IF_CONSTEXPR) && __cplusplus >= 201703L
#define SIMD_IF_CONSTEXPR if constexpr
#endif

#if !defined(SIMD_IF_CONSTEXPR)
#define SIMD_IF_CONSTEXPR if
#endif

#include "simd/config/config.h"
#include "simd/config/inline.h"
#include "simd/memory/aligned_allocator.h"

#include "simd/types/vec.h"
#include "simd/types/traits.h"
#include "simd/types/api.h"
