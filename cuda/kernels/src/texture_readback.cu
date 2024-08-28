#include <cuda_runtime.h>

#include <numeric>

__device__ inline ushort4 rgbaTo16bit(float4 rgba)
{
    constexpr uint16_t max16bitValue = 65535;
    constexpr float scale = max16bitValue;

    return ushort4 {
        static_cast<uint16_t>(__float2uint_rn(rgba.x * scale)),
        static_cast<uint16_t>(__float2uint_rn(rgba.y * scale)),
        static_cast<uint16_t>(__float2uint_rn(rgba.z * scale)),
        static_cast<uint16_t>(__float2uint_rn(rgba.w * scale))
    };
}

extern "C" __global__ void readback_texture(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
    )
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width || y >= height)
    {
        return;
    }

    float4 rgba = tex2D<float4>(textureInput, x, y);

    ushort4* outPtr = static_cast<ushort4*>(bufferOutput);

    outPtr[y * width + x] = rgbaTo16bit(rgba);
}

template<size_t xBatchSize>
__device__ inline void readback_texture_batched(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
)
{
    size_t xBatch = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    size_t x = xBatch * xBatchSize;

    if(x >= width || y >= height)
    {
        return;
    }

    //size_t count = min(width - x, xBatchSize);

    float4 rgba[xBatchSize];

#pragma unroll
    for (size_t xOffset = 0; xOffset < xBatchSize; ++xOffset)
    {
        //if(xOffset >= count)
        //    break;
        rgba[xOffset] = tex2D<float4>(textureInput, x + xOffset, y);
    }

    ushort4* outPtr = static_cast<ushort4*>(bufferOutput);

#pragma unroll
    for (size_t xOffset = 0; xOffset < xBatchSize; ++xOffset)
    {
        //if(xOffset >= count)
        //    break;
        outPtr[y * width + x + xOffset] = rgbaTo16bit(rgba[xOffset]);
    }
}

extern "C" __global__ void readback_texture_batched_2(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
)
{
    readback_texture_batched<2>(width, height, textureInput, bufferOutput);
}

extern "C" __global__ void readback_texture_batched_4(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
)
{
    readback_texture_batched<4>(width, height, textureInput, bufferOutput);
}

extern "C" __global__ void readback_texture_batched_8(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
)
{
    readback_texture_batched<8>(width, height, textureInput, bufferOutput);
}

extern "C" __global__ void readback_texture_batched_16(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
)
{
    readback_texture_batched<16>(width, height, textureInput, bufferOutput);
}

extern "C" __global__ void readback_texture_batched_32(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput
)
{
    readback_texture_batched<32>(width, height, textureInput, bufferOutput);
}

extern "C" __global__ void readback_texture_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width || y >= height)
    {
        return;
    }

    float4 rgba = tex2D<float4>(textureInput, x, y);

    ushort4* outPtr = reinterpret_cast<ushort4*>(static_cast<uint8_t*>(bufferOutput) + y * outputPitch);

    outPtr[x] = rgbaTo16bit(rgba);
}

template<size_t xBatchSize>
__device__ inline void readback_texture_batched_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    size_t xBatch = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    size_t x = xBatch * xBatchSize;

    if(x >= width || y >= height)
    {
        return;
    }

    //size_t count = min(width - x, xBatchSize);

    float4 rgba[xBatchSize];

#pragma unroll
    for (size_t xOffset = 0; xOffset < xBatchSize; ++xOffset)
    {
        //if(xOffset >= count)
        //    break;
        rgba[xOffset] = tex2D<float4>(textureInput, x + xOffset, y);
    }

    ushort4* outPtr = reinterpret_cast<ushort4*>(static_cast<uint8_t*>(bufferOutput) + y * outputPitch);

#pragma unroll
    for (size_t xOffset = 0; xOffset < xBatchSize; ++xOffset)
    {
        //if(xOffset >= count)
        //    break;
        outPtr[x + xOffset] = rgbaTo16bit(rgba[xOffset]);
    }
}

extern "C" __global__ void readback_texture_batched_2_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    readback_texture_batched_pitched_output<2>(width, height, textureInput, bufferOutput, outputPitch);
}

extern "C" __global__ void readback_texture_batched_4_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    readback_texture_batched_pitched_output<4>(width, height, textureInput, bufferOutput, outputPitch);
}

extern "C" __global__ void readback_texture_batched_8_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    readback_texture_batched_pitched_output<8>(width, height, textureInput, bufferOutput, outputPitch);
}

extern "C" __global__ void readback_texture_batched_16_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    readback_texture_batched_pitched_output<16>(width, height, textureInput, bufferOutput, outputPitch);
}

extern "C" __global__ void readback_texture_batched_32_pitched_output(
    size_t width, size_t height,
    cudaTextureObject_t textureInput,
    void* bufferOutput, size_t outputPitch
)
{
    readback_texture_batched_pitched_output<32>(width, height, textureInput, bufferOutput, outputPitch);
}