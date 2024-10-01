#pragma once

#include <conversions.cuh>
#include <yuv.cuh>

constexpr uint32_t uyvy10beBlockSize = 8;
constexpr uint32_t uyvy10leBlockSize = 2;
constexpr size_t uyvy10_bits = 10;

__device__ inline ushort4
convertRgbaFloatToUYVY10bit(const float4 * rgba, color_space colorSpace,
                            bool fullRange, bool preserveRangeOvershoot)
{
    ushort4 uyvy;

    float3 yuv0 = YUVfromRGB<false>(float3{rgba[0].x, rgba[0].y, rgba[0].z}, colorSpace);
    float y1 = YUVfromRGB<true>(float3{rgba[0].x, rgba[0].y, rgba[0].z}, colorSpace);

    uyvy.x = fromYUVFloat<uyvy10_bits, true>(yuv0.y, fullRange, preserveRangeOvershoot);  // U
    uyvy.y = fromYUVFloat<uyvy10_bits, false>(yuv0.x, fullRange, preserveRangeOvershoot);  // Y0
    uyvy.z = fromYUVFloat<uyvy10_bits, true>(yuv0.z, fullRange, preserveRangeOvershoot);  // V
    uyvy.w = fromYUVFloat<uyvy10_bits, false>(y1, fullRange, preserveRangeOvershoot);  // Y1

    return uyvy;
}

__device__ inline void
encode2pxUYVY10bitToUYVY10BE(uint8_t * uyvy10be, ushort4 uyvy)
{
    // U Y0 V Y1 (10-bits per component)
    uyvy10be[0] =                  uyvy.x >> 2;
    uyvy10be[1] = (uyvy.x << 6) | (uyvy.y >> 4);
    uyvy10be[2] = (uyvy.y << 4) | (uyvy.z >> 6);
    uyvy10be[3] = (uyvy.z << 2) | (uyvy.w >> 8);
    uyvy10be[4] = (uyvy.w);
}

__device__ inline void
encode2pxUYVY10bitToUYVY10LE(uint8_t * uyvy10le, ushort4 uyvy)
{
    // V Y1 U Y0 (10-bits per component)
    uyvy10le[0] =                     uyvy.z;
    uyvy10le[1] = (uyvy.z >> 8) | (uyvy.w << 2);
    uyvy10le[2] = (uyvy.w >> 6) | (uyvy.x << 4);
    uyvy10le[3] = (uyvy.x >> 4) | (uyvy.y << 6);
    uyvy10le[4] =  uyvy.y >> 2;
}

__global__ void kernel_uyvy10be_output_single_pass(
    cudaTextureObject_t src,
    uint8_t * destUYVY,
    size_t pitchUYVY,
    uint32_t width, uint32_t height,
    color_space colorSpace, bool fullRange, bool preserveRangeOvershoot)
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    x *= uyvy10beBlockSize;

    if (x >= width || y >= height)
        return;

    constexpr size_t outputBytesPerThread = uyvy10beBlockSize / 2 * 5;

    alignas(max_align_t) uint8_t storeUYVY10BE[outputBytesPerThread];
    #pragma unroll
    for (int i = 0; i < uyvy10beBlockSize / 2; i++)
    {
        const float4 rgba[2] = {
            unpremultiply(tex2D<float4>(src, x + 2 * i + 0, y)),
            unpremultiply(tex2D<float4>(src, x + 2 * i + 1, y))
        };

        const auto uyvy = convertRgbaFloatToUYVY10bit(rgba, colorSpace, fullRange, preserveRangeOvershoot);

        encode2pxUYVY10bitToUYVY10BE(storeUYVY10BE, uyvy);
    }

    const size_t offsetUYVY = y * pitchUYVY + threadIdx.x * outputBytesPerThread;
    memcpy<outputBytesPerThread>(destUYVY + offsetUYVY, storeUYVY10BE);
}

size_t calculateRequiredSharedMemoryUYVY10BE(uint3 blockDim)
{
    const size_t uyvy10beSharedPitch = blockDim.x * uyvy10beBlockSize / 2 * sizeof(uint64_t);
    return blockDim.y * uyvy10beSharedPitch;
}

__global__ void kernel_uyvy10be_output_single_pass_shared(
    cudaTextureObject_t src,
    uint8_t * destUYVY,
    size_t pitchUYVY,
    uint32_t width, uint32_t height,
    color_space colorSpace, bool fullRange, bool preserveRangeOvershoot)
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    x *= uyvy10beBlockSize;

    if (x >= width || y >= height)
        return;

    constexpr size_t outputBytesPerThread = uyvy10beBlockSize / 2 * 5;

    alignas(max_align_t) uint8_t storeUYVY10BE[outputBytesPerThread];
    #pragma unroll
    for (int i = 0; i < uyvy10beBlockSize / 2; i++)
    {
        const float4 rgba[2] = {
            unpremultiply(tex2D<float4>(src, x + 2 * i + 0, y)),
            unpremultiply(tex2D<float4>(src, x + 2 * i + 1, y))
        };

        const auto uyvy = convertRgbaFloatToUYVY10bit(rgba, colorSpace, fullRange, preserveRangeOvershoot);

        encode2pxUYVY10bitToUYVY10BE(storeUYVY10BE, uyvy);
    }

    extern __shared__ uint8_t sharedUYVY10BE[];
    const size_t uyvy10beSharedPitch = blockDim.x * uyvy10beBlockSize / 2 * sizeof(uint64_t);
    const size_t uyvy10beBlockXBytes = blockDim.x * outputBytesPerThread;

    const size_t offsetYShared = threadIdx.y * uyvy10beSharedPitch;
    const size_t offsetXSharedWrite = threadIdx.x * outputBytesPerThread;

    memcpy<outputBytesPerThread>(sharedUYVY10BE + offsetYShared + offsetXSharedWrite, storeUYVY10BE);

    __syncthreads();

    #define UYVY_WRITE_GLOBAL_OPTIMIZED(Type) \
    if (uyvy10beBlockXBytes % sizeof(Type) == 0) \
    { \
        const size_t maxThreadsX = uyvy10beBlockXBytes / sizeof(Type); \
        if(threadIdx.x >= maxThreadsX) \
            return; \
        const size_t offsetXSharedRead = threadIdx.x * sizeof(Type); \
        const size_t offsetUYVYGlobalWrite = y * pitchUYVY + threadIdx.x * sizeof(Type); \
        memcpy<sizeof(Type)>(destUYVY + offsetUYVYGlobalWrite, sharedUYVY10BE + offsetYShared + offsetXSharedRead); \
        return; \
    }

    UYVY_WRITE_GLOBAL_OPTIMIZED(ulonglong4)
    UYVY_WRITE_GLOBAL_OPTIMIZED(ulonglong2)
    UYVY_WRITE_GLOBAL_OPTIMIZED(ulonglong1)
    UYVY_WRITE_GLOBAL_OPTIMIZED(uint1)
    UYVY_WRITE_GLOBAL_OPTIMIZED(ushort1)
    UYVY_WRITE_GLOBAL_OPTIMIZED(uchar1)
}
