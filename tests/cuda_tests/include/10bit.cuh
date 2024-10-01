#pragma once

#include <conversions.cuh>
#include <yuv.cuh>

constexpr uint32_t yuv444p10leBlockSize = 2;
constexpr uint32_t yuv422p10leBlockSize = 2;
constexpr uint32_t yuv420p10leBlockSize = 2;
using yuv10bit_t = uint16_t;
constexpr size_t yuv10bit_bits = 8;

__global__ void kernel_yuv444p10le_output_single_pass(
    cudaTextureObject_t src,
    uint8_t * destY,
    uint8_t * destU,
    uint8_t * destV,
    size_t pitchY,
    size_t pitchU,
    size_t pitchV,
    uint32_t width, uint32_t height,
    color_space colorSpace, bool fullRange, bool preserveRangeOvershoot)
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    x *= yuv444p10leBlockSize;

    if (x >= width || y >= height)
        return;

    alignas(max_align_t) yuv10bit_t storeY[yuv444p10leBlockSize];
    alignas(max_align_t) yuv10bit_t storeU[yuv444p10leBlockSize];
    alignas(max_align_t) yuv10bit_t storeV[yuv444p10leBlockSize];
    #pragma unroll
    for (int i = 0; i < yuv444p10leBlockSize; i++)
    {
        const auto rgba = unpremultiply(tex2D<float4>(src, x + i, y));
        const auto yuv = YUVfromRGB<false>(float3{rgba.x, rgba.y, rgba.z}, colorSpace);

        storeY[i] = fromYUVFloat<yuv10bit_bits, false>(yuv.x, fullRange, preserveRangeOvershoot);
        storeU[i] = fromYUVFloat<yuv10bit_bits, true>(yuv.y, fullRange, preserveRangeOvershoot);
        storeV[i] = fromYUVFloat<yuv10bit_bits, true>(yuv.z, fullRange, preserveRangeOvershoot);
    }

    auto offsetY = y * pitchY + x * sizeof(yuv10bit_t);
    auto offsetU = y * pitchU + x * sizeof(yuv10bit_t);
    auto offsetV = y * pitchV + x * sizeof(yuv10bit_t);

    memcpy<yuv444p10leBlockSize * sizeof(yuv10bit_t)>(destY + offsetY, storeY);
    memcpy<yuv444p10leBlockSize * sizeof(yuv10bit_t)>(destU + offsetU, storeU);
    memcpy<yuv444p10leBlockSize * sizeof(yuv10bit_t)>(destV + offsetV, storeV);
}

__global__ void kernel_yuv422p10le_output_single_pass(
    cudaTextureObject_t src,
    uint8_t * destY,
    uint8_t * destU,
    uint8_t * destV,
    size_t pitchY,
    size_t pitchU,
    size_t pitchV,
    uint32_t width, uint32_t height,
    color_space colorSpace, bool fullRange, bool preserveRangeOvershoot)
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    x *= yuv422p10leBlockSize;

    if (x >= width || y >= height)
        return;

    alignas(max_align_t) yuv10bit_t storeY[yuv422p10leBlockSize];
    alignas(max_align_t) yuv10bit_t storeU[yuv422p10leBlockSize / 2];
    alignas(max_align_t) yuv10bit_t storeV[yuv422p10leBlockSize / 2];
    #pragma unroll
    for(int i = 0; i < yuv422p10leBlockSize; i += 2)
    {
        const auto rgba0 = unpremultiply(tex2D<float4>(src, x + 2 * i + 0, y));
        const auto rgba1 = unpremultiply(tex2D<float4>(src, x + 2 * i + 1, y));
        const auto yuv0 = YUVfromRGB<false>(float3{rgba0.x, rgba0.y, rgba0.z}, colorSpace);
        const auto yuv1 = YUVfromRGB<true>(float3{rgba1.x, rgba1.y, rgba1.z}, colorSpace);

        storeY[2 * i + 0] = fromYUVFloat<yuv10bit_bits, false>(yuv0.x, fullRange, preserveRangeOvershoot);
        storeY[2 * i + 1] = fromYUVFloat<yuv10bit_bits, false>(yuv1, fullRange, preserveRangeOvershoot);
        storeU[i] = fromYUVFloat<yuv10bit_bits, true>(yuv0.y, fullRange, preserveRangeOvershoot);
        storeV[i] = fromYUVFloat<yuv10bit_bits, true>(yuv0.z, fullRange, preserveRangeOvershoot);
    }

    auto offsetY = y * pitchY + x * sizeof(yuv10bit_t);
    auto offsetU = y * pitchU + x / 2 * sizeof(yuv10bit_t);
    auto offsetV = y * pitchV + x / 2 * sizeof(yuv10bit_t);

    memcpy<yuv422p10leBlockSize * sizeof(yuv10bit_t)>(destY + offsetY, storeY);
    memcpy<yuv422p10leBlockSize / 2 * sizeof(yuv10bit_t)>(destU + offsetU, storeU);
    memcpy<yuv422p10leBlockSize / 2 * sizeof(yuv10bit_t)>(destV + offsetV, storeV);
}

__global__ void kernel_yuv420p10le_output_single_pass(
    cudaTextureObject_t src,
    uint8_t * destY,
    uint8_t * destU,
    uint8_t * destV,
    size_t pitchY,
    size_t pitchU,
    size_t pitchV,
    uint32_t width, uint32_t height,
    color_space colorSpace, bool fullRange, bool preserveRangeOvershoot)
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    x *= yuv420p10leBlockSize;

    if(x >= width || y >= height)
        return;

    if(y % 2 == 0)
    {
        alignas(max_align_t) yuv10bit_t storeY[yuv420p10leBlockSize];
        alignas(max_align_t) yuv10bit_t storeU[yuv420p10leBlockSize / 2];
        alignas(max_align_t) yuv10bit_t storeV[yuv420p10leBlockSize / 2];
        #pragma unroll
        for(int i = 0; i < yuv420p10leBlockSize; i += 2)
        {
            const auto rgba0 = unpremultiply(tex2D<float4>(src, x + 2 * i + 0, y));
            const auto rgba1 = unpremultiply(tex2D<float4>(src, x + 2 * i + 1, y));
            const auto yuv0 = YUVfromRGB<false>(float3{rgba0.x, rgba0.y, rgba0.z}, colorSpace);
            const auto yuv1 = YUVfromRGB<true>(float3{rgba1.x, rgba1.y, rgba1.z}, colorSpace);

            storeY[2 * i + 0] = fromYUVFloat<yuv10bit_bits, false>(yuv0.x, fullRange, preserveRangeOvershoot);
            storeY[2 * i + 1] = fromYUVFloat<yuv10bit_bits, false>(yuv1, fullRange, preserveRangeOvershoot);
            storeU[i] = fromYUVFloat<yuv10bit_bits, true>(yuv0.y, fullRange, preserveRangeOvershoot);
            storeV[i] = fromYUVFloat<yuv10bit_bits, true>(yuv0.z, fullRange, preserveRangeOvershoot);
        }

        auto offsetY = y * pitchY + x * sizeof(yuv10bit_t);
        auto offsetU = (y / 2) * pitchU + x / 2 * sizeof(yuv10bit_t);
        auto offsetV = (y / 2) * pitchV + x / 2 * sizeof(yuv10bit_t);

        memcpy<yuv420p10leBlockSize * sizeof(yuv10bit_t)>(destY + offsetY, storeY);
        memcpy<yuv420p10leBlockSize / 2 * sizeof(yuv10bit_t)>(destU + offsetU, storeU);
        memcpy<yuv420p10leBlockSize / 2 * sizeof(yuv10bit_t)>(destV + offsetV, storeV);
    }
    else
    {
        alignas(max_align_t) yuv10bit_t storeY[yuv420p10leBlockSize];
        #pragma unroll
        for(int i = 0; i < yuv420p10leBlockSize; i++)
        {
            const auto rgba = unpremultiply(tex2D<float4>(src, x + i, y));
            const auto yuv = YUVfromRGB<true>(float3{rgba.x, rgba.y, rgba.z}, colorSpace);
            storeY[i] = fromYUVFloat<yuv10bit_bits, false>(yuv, fullRange, preserveRangeOvershoot);
        }

        auto offsetY = y * pitchY + x * sizeof(yuv10bit_t);

        memcpy<yuv420p10leBlockSize * sizeof(yuv10bit_t)>(destY + offsetY, storeY);
    }
}
