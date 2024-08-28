#include <cuda_runtime.h>

using uchar = uint8_t;
using uint = uint32_t;

__device__ inline float4 read_raw_R8G8B8A8_normalized(const void* in, size_t pixelOffset)
{
    const auto dataInBegin = reinterpret_cast<const uchar*>(in) + pixelOffset * sizeof(uchar4);

    const auto r = dataInBegin[0];
    const auto g = dataInBegin[1];
    const auto b = dataInBegin[2];
    const auto a = dataInBegin[3];

    return make_float4(
        static_cast<float>(r) / 255.0f,
        static_cast<float>(g) / 255.0f,
        static_cast<float>(b) / 255.0f,
        static_cast<float>(a) / 255.0f
    );
}

__device__ inline void write_raw_R8G8B8A8_normalized(void* out, size_t pixelOffset, float4 value)
{
    const auto dataOutBegin = reinterpret_cast<uchar*>(out) + pixelOffset * sizeof(uchar4);

    dataOutBegin[0] = static_cast<uchar>(value.x * 255.0f);
    dataOutBegin[1] = static_cast<uchar>(value.y * 255.0f);
    dataOutBegin[2] = static_cast<uchar>(value.z * 255.0f);
    dataOutBegin[3] = static_cast<uchar>(value.w * 255.0f);
}

__device__ inline float3 read_lut3D(cudaTextureObject_t lut, size_t x, size_t y, size_t z)
{
    auto rgbx = tex3D<float4>(lut, x, y, z);
    return float3{
        .x = rgbx.x,
        .y = rgbx.y,
        .z = rgbx.z,
    };
}

__device__ float3 interpolate_tetrahedral(float3 inRGB, cudaTextureObject_t lut3D, size_t lutSize)
{
    inRGB = float3 {
        inRGB.x * static_cast<float>(lutSize - 1),
        inRGB.y * static_cast<float>(lutSize - 1),
        inRGB.z * static_cast<float>(lutSize - 1),
    };

    const auto prev = uint3{
        .x = __float2uint_rd(inRGB.x),
        .y = __float2uint_rd(inRGB.y),
        .z = __float2uint_rd(inRGB.z),
    };
    const auto next = uint3{
        .x = __float2uint_ru(inRGB.x),
        .y = __float2uint_ru(inRGB.y),
        .z = __float2uint_ru(inRGB.z),
    };

    const auto d = float3{
        .x = inRGB.x - static_cast<float>(prev.x),
        .y = inRGB.y - static_cast<float>(prev.y),
        .z = inRGB.z - static_cast<float>(prev.z),
    };

    const auto c000 = read_lut3D(lut3D, prev.x, prev.y, prev.z);
    const auto c111 = read_lut3D(lut3D, next.x, next.y, next.z);
    float3 c;
    if (d.x > d.y) {
        if (d.y > d.z) {
            const auto c100 = read_lut3D(lut3D, next.x, prev.y, prev.z);
            const auto c110 = read_lut3D(lut3D, next.x, next.y, prev.z);
            c.x = (1-d.x) * c000.x + (d.x-d.y) * c100.x + (d.y-d.z) * c110.x + (d.z) * c111.x;
            c.y = (1-d.x) * c000.y + (d.x-d.y) * c100.y + (d.y-d.z) * c110.y + (d.z) * c111.y;
            c.z = (1-d.x) * c000.z + (d.x-d.y) * c100.z + (d.y-d.z) * c110.z + (d.z) * c111.z;
        } else if (d.x > d.z) {
            const auto c100 = read_lut3D(lut3D, next.x, prev.y, prev.z);
            const auto c101 = read_lut3D(lut3D, next.x, prev.y, next.z);
            c.x = (1-d.x) * c000.x + (d.x-d.z) * c100.x + (d.z-d.y) * c101.x + (d.y) * c111.x;
            c.y = (1-d.x) * c000.y + (d.x-d.z) * c100.y + (d.z-d.y) * c101.y + (d.y) * c111.y;
            c.z = (1-d.x) * c000.z + (d.x-d.z) * c100.z + (d.z-d.y) * c101.z + (d.y) * c111.z;
        } else {
            const auto c001 = read_lut3D(lut3D, prev.x, prev.y, next.z);
            const auto c101 = read_lut3D(lut3D, next.x, prev.y, next.z);
            c.x = (1-d.z) * c000.x + (d.z-d.x) * c001.x + (d.x-d.y) * c101.x + (d.y) * c111.x;
            c.y = (1-d.z) * c000.y + (d.z-d.x) * c001.y + (d.x-d.y) * c101.y + (d.y) * c111.y;
            c.z = (1-d.z) * c000.z + (d.z-d.x) * c001.z + (d.x-d.y) * c101.z + (d.y) * c111.z;
        }
    } else {
        if (d.z > d.y) {
            const auto c001 = read_lut3D(lut3D, prev.x, prev.y, next.z);
            const auto c011 = read_lut3D(lut3D, prev.x, next.y, next.z);
            c.x = (1-d.z) * c000.x + (d.z-d.y) * c001.x + (d.y-d.x) * c011.x + (d.x) * c111.x;
            c.y = (1-d.z) * c000.y + (d.z-d.y) * c001.y + (d.y-d.x) * c011.y + (d.x) * c111.y;
            c.z = (1-d.z) * c000.z + (d.z-d.y) * c001.z + (d.y-d.x) * c011.z + (d.x) * c111.z;
        } else if (d.z > d.x) {
            const auto c010 = read_lut3D(lut3D, prev.x, next.y, prev.z);
            const auto c011 = read_lut3D(lut3D, prev.x, next.y, next.z);
            c.x = (1-d.y) * c000.x + (d.y-d.z) * c010.x + (d.z-d.x) * c011.x + (d.x) * c111.x;
            c.y = (1-d.y) * c000.y + (d.y-d.z) * c010.y + (d.z-d.x) * c011.y + (d.x) * c111.y;
            c.z = (1-d.y) * c000.z + (d.y-d.z) * c010.z + (d.z-d.x) * c011.z + (d.x) * c111.z;
        } else {
            const auto c010 = read_lut3D(lut3D, prev.x, next.y, prev.z);
            const auto c110 = read_lut3D(lut3D, next.x, next.y, prev.z);
            c.x = (1-d.y) * c000.x + (d.y-d.x) * c010.x + (d.x-d.z) * c110.x + (d.z) * c111.x;
            c.y = (1-d.y) * c000.y + (d.y-d.x) * c010.y + (d.x-d.z) * c110.y + (d.z) * c111.y;
            c.z = (1-d.y) * c000.z + (d.y-d.x) * c010.z + (d.x-d.z) * c110.z + (d.z) * c111.z;
        }
    }
    return c;
}

extern "C" __global__ void apply_lut3D_R8G8B8A8_1D_raw_in_raw_out(
    const void* in,
    void* out,
    size_t size,
    cudaTextureObject_t lut3D,
    size_t lutSize
)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size)
        return;

    const auto inRGBA = read_raw_R8G8B8A8_normalized(in, index);
    const auto inRGB = make_float3(inRGBA.x, inRGBA.y, inRGBA.z);
    const auto outRGB = interpolate_tetrahedral(inRGB, lut3D, lutSize);
    const auto outRGBA = make_float4(outRGB.x, outRGB.y, outRGB.z, inRGBA.w);

    write_raw_R8G8B8A8_normalized(out, index, outRGBA);
}

extern "C" __global__ void apply_lut3D_R8G8B8A8_2D_raw_in_raw_out(
    void* in,
    void* out,
    size_t width,
    size_t height,
    cudaTextureObject_t lut3D,
    size_t lutSize
)
{
    const size_t indexX = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t indexY = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexX >= width || indexY >= height)
        return;

    const size_t index = indexY * width + indexX;

    const auto inRGBA = read_raw_R8G8B8A8_normalized(in, index);
    const auto inRGB = make_float3(inRGBA.x, inRGBA.y, inRGBA.z);
    const auto outRGB = interpolate_tetrahedral(inRGB, lut3D, lutSize);
    const auto outRGBA = make_float4(outRGB.x, outRGB.y, outRGB.z, inRGBA.w);

    write_raw_R8G8B8A8_normalized(out, index, outRGBA);
}

extern "C" __global__ void apply_lut3D_R8G8B8A8_2D_texture_in_raw_out(
    cudaTextureObject_t in,
    void* out,
    size_t width,
    size_t height,
    cudaTextureObject_t lut3D,
    size_t lutSize
)
{
    const size_t indexX = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t indexY = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexX >= width || indexY >= height)
        return;

    const size_t index = indexY * width + indexX;

    const auto inRGBA = tex2D<float4>(in, indexX, indexY);
    const auto inRGB = make_float3(inRGBA.x, inRGBA.y, inRGBA.z);
    const auto outRGB = interpolate_tetrahedral(inRGB, lut3D, lutSize);
    const auto outRGBA = make_float4(outRGB.x, outRGB.y, outRGB.z, inRGBA.w);

    write_raw_R8G8B8A8_normalized(out, index, outRGBA);
}

extern "C" __global__ void apply_lut3D_R8G8B8A8_2D_texture_in_surface_out(
    cudaTextureObject_t in,
    cudaSurfaceObject_t out,
    size_t width,
    size_t height,
    cudaTextureObject_t lut3D,
    size_t lutSize
)
{
    const size_t indexX = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t indexY = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexX >= width || indexY >= height)
        return;

    const auto inRGBA = tex2D<float4>(in, indexX, indexY);
    const auto inRGB = make_float3(inRGBA.x, inRGBA.y, inRGBA.z);
    const auto outRGB = interpolate_tetrahedral(inRGB, lut3D, lutSize);
    const auto outRGBA = make_float4(outRGB.x, outRGB.y, outRGB.z, inRGBA.w);

    const auto outR8G8B8A8 = uchar4{
        static_cast<uchar>(outRGBA.x * 255.0f),
        static_cast<uchar>(outRGBA.y * 255.0f),
        static_cast<uchar>(outRGBA.z * 255.0f),
        static_cast<uchar>(outRGBA.w * 255.0f),
    };
    surf2Dwrite(outR8G8B8A8, out, indexX * 4, indexY);
}