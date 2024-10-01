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

template<typename Type, size_t Count>
__device__ inline void copy(void* dst, const void* src)
{
    auto* d = reinterpret_cast<Type*>(dst);
    const auto* s = reinterpret_cast<const Type*>(src);
    #pragma unroll
    for(size_t i=0;i<Count;i++)
    {
        d[i] = s[i];
    }
}

template<typename Type>
__device__ inline void copy(void* dst, const void* src, size_t count)
{
    auto* d = reinterpret_cast<Type*>(dst);
    const auto* s = reinterpret_cast<const Type*>(src);
    #pragma unroll
    for(size_t i=0;i<count;i++)
    {
        d[i] = s[i];
    }
}

template<size_t CountBytes>
__device__ inline void memcpy(void* dst, const void* src)
{
    #define TRY_COPY(Type) \
    if constexpr (CountBytes % sizeof(Type) == 0) \
    { \
        copy<Type, CountBytes / sizeof(Type)>(dst, src); \
        return; \
    }

    TRY_COPY(ulonglong4)
    TRY_COPY(ulonglong2)
    TRY_COPY(ulonglong1)
    TRY_COPY(uint1)
    TRY_COPY(ushort1)
    TRY_COPY(uchar1)

    #undef TRY_COPY
}

__device__ inline void memcpyE(void* dst, const void* src, size_t byteCount)
{
    #define TRY_COPY(Type) \
    if (byteCount % sizeof(Type) == 0) \
    { \
        copy<Type>(dst, src, byteCount / sizeof(Type)); \
        return; \
    }

    TRY_COPY(ulonglong4)
    TRY_COPY(ulonglong2)
    TRY_COPY(ulonglong1)
    TRY_COPY(uint1)
    TRY_COPY(ushort1)
    TRY_COPY(uchar1)

    #undef TRY_COPY
}
