#pragma once

#include <common.cuh>

template<color_space ColorSpace>
struct ycbcr { };

template<>
struct ycbcr<color_space::rec601>
{
private:
    constexpr static float Kyr601f = 0.299f;
    constexpr static float Kyg601f = 0.587f;
    constexpr static float Kyb601f = 0.114f;
    constexpr static float Kcb601f = 1.772f;
    constexpr static float Kcr601f = 1.402f;
public:
    template<bool OnlyLuminance>
    __host__ __device__ static inline auto fromRGB(float3 rgb)
    {
        float y = Kyr601f * rgb.x + Kyg601f * rgb.y + Kyb601f * rgb.z;
        if constexpr (OnlyLuminance)
        {
            return y;
        }
        else
        {
            return float3
                {
                    y,
                    (rgb.y - y) / Kcb601f,
                    (rgb.z - y) / Kcr601f,
                };
        }
    }

    __host__ __device__ static inline float3 toRGB(float3 yuv)
    {
        return float3
            {
                yuv.x + Kcr601f * yuv.z,
                yuv.x + Kcb601f * yuv.y,
                yuv.x - (Kyr601f * Kcr601f * yuv.z + Kyb601f * Kcb601f * yuv.y) / Kyg601f,
            };
    }
};

template<>
struct ycbcr<color_space::rec709>
{
private:
    constexpr static float Kyr709f = 0.2126f;
    constexpr static float Kyg709f = 0.7152f;
    constexpr static float Kyb709f = 0.0722f;
    constexpr static float Kcb709f = 1.8556f;
    constexpr static float Kcr709f = 1.5748f;
public:
    template<bool OnlyLuminance>
    __host__ __device__ static inline auto fromRGB(float3 rgb)
    {
        float y = Kyr709f * rgb.x + Kyg709f * rgb.y + Kyb709f * rgb.z;
        if constexpr (OnlyLuminance)
        {
            return y;
        }
        else
        {
            return float3
                {
                    y,
                    (rgb.y - y) / Kcb709f,
                    (rgb.z - y) / Kcr709f,
                };
        }
    }

    __host__ __device__ static inline float3 toRGB(float3 yuv)
    {
        return float3
            {
                yuv.x + Kcr709f * yuv.z,
                yuv.x + Kcb709f * yuv.y,
                yuv.x - (Kyr709f * Kcr709f * yuv.z + Kyb709f * Kcb709f * yuv.y) / Kyg709f,
            };
    }
};

template<>
struct ycbcr<color_space::rec2020>
{
private:
    constexpr static float Kyr2020f = 0.2627f;
    constexpr static float Kyg2020f = 0.6780f;
    constexpr static float Kyb2020f = 0.0593f;
    constexpr static float Kcb2020f = 1.8814f;
    constexpr static float Kcr2020f = 1.4746f;
public:
    template<bool OnlyLuminance>
    __host__ __device__ static inline auto fromRGB(float3 rgb)
    {
        float y = Kyr2020f * rgb.x + Kyg2020f * rgb.y + Kyb2020f * rgb.z;
        if constexpr (OnlyLuminance)
        {
            return y;
        }
        else
        {
            return float3
                {
                    y,
                    (rgb.y - y) / Kcb2020f,
                    (rgb.z - y) / Kcr2020f,
                };
        }
    }

    __host__ __device__ static inline float3 toRGB(float3 yuv)
    {
        return float3
            {
                yuv.x + Kcr2020f * yuv.z,
                yuv.x + Kcb2020f * yuv.y,
                yuv.x - (Kyr2020f * Kcr2020f * yuv.z + Kyb2020f * Kcb2020f * yuv.y) / Kyg2020f,
            };
    }
};

template<bool OnlyLuminance>
__host__ __device__ static inline auto YUVfromRGB(float3 rgb, color_space colorSpace)
{
    switch (colorSpace)
    {
        case color_space::rec601:
            return ycbcr<color_space::rec601>::fromRGB<OnlyLuminance>(rgb);
        case color_space::rec709:
            return ycbcr<color_space::rec709>::fromRGB<OnlyLuminance>(rgb);
        case color_space::rec2020:
            return ycbcr<color_space::rec2020>::fromRGB<OnlyLuminance>(rgb);
    }

    if constexpr (OnlyLuminance)
    {
        return float{};
    }
    else
    {
        return float3{};
    }
}

__host__ __device__ static inline float3 RGBfromYUV(float3 rgb, color_space colorSpace)
{
    switch (colorSpace)
    {
        case color_space::rec601:
            return ycbcr<color_space::rec601>::toRGB(rgb);
        case color_space::rec709:
            return ycbcr<color_space::rec709>::toRGB(rgb);
        case color_space::rec2020:
            return ycbcr<color_space::rec2020>::toRGB(rgb);
    }

    return float3{};
}