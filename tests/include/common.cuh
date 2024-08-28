#pragma once

#include <numeric>
#include <cstdint>
#include <cassert>

void cudaAssert(cudaError_t error)
{
    if(error != cudaSuccess)
    {
        assert(error == cudaSuccess);
    }
}

enum class color_space
{
    rec601,
    rec709,
    rec2020
};

enum class color_transfer_function
{
    linear,
    gamma22,
    srgb,
    rec1886,
    pq,
    hlg
};

template<typename Type>
__device__ __host__ static inline Type clamp(Type clampMin, Type clampMax, Type value)
{
    return min(clampMax, max(clampMin, value));
}

__device__ __host__ static inline float4 unpremultiply(float4 rgba)
{
    float inverseAlpha = 1.0f / rgba.w;
    return {
        fminf(rgba.x * inverseAlpha, 1.0f),
        fminf(rgba.y * inverseAlpha, 1.0f),
        fminf(rgba.z * inverseAlpha, 1.0f),
        rgba.w,
    };
}

template<size_t CountBytes>
__device__ inline void memcpyEfficient(void* dst, const void* src)
{
    if constexpr (CountBytes % 4 == 0)
    {
        auto* dst32 = reinterpret_cast<uint32_t*>(dst);
        const auto* src32 = reinterpret_cast<const uint32_t*>(src);
        #pragma unroll
        for (int i = 0; i < CountBytes / 4; i++)
        {
            dst32[i] = src32[i];
        }
    }
    else if constexpr (CountBytes % 2 == 0)
    {
        auto* dst16 = reinterpret_cast<uint16_t*>(dst);
        const auto* src16 = reinterpret_cast<const uint16_t*>(src);
        #pragma unroll
        for (int i = 0; i < CountBytes / 2; i++)
        {
            dst16[i] = src16[i];
        }
    }
    else
    {
        memcpy(dst, src, CountBytes);
    }
}
