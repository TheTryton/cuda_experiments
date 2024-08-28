#pragma once

#include <common.cuh>

template<color_space ColorSpace, color_transfer_function ColorTransferFunction>
struct convert { };

template<size_t Bits>
struct quantization
{ };

template<>
struct quantization<8>
{
    struct rgb
    {
    private:
        constexpr static float scale = 255.0f;
    public:
        __host__ __device__ static inline uint8_t fromFloat(float value)
        {
            #ifdef  __CUDA_ARCH__
            return __float2uint_rn(value * scale);
            #else
            return std::lround(value * scale);
            #endif
        }
        __host__ __device__ static inline float toFloat(uint8_t value)
        {
            #ifdef  __CUDA_ARCH__
            return __uint2float_rn(value) / scale;
            #else
            return static_cast<float>(value) * scale;
            #endif
        }
    };

    struct yuv
    {
    private:
        constexpr static uint8_t minY = 16;
        constexpr static uint8_t maxY = 235;
        constexpr static uint8_t minC = 16;
        constexpr static uint8_t maxC = 240;
        constexpr static uint8_t fullMin = 0;
        constexpr static uint8_t fullMax = 255;
        constexpr static uint8_t legalMin = 1;
        constexpr static uint8_t legalMax = 254;
        constexpr static uint8_t zeroC = 128;

        template<bool Chroma, bool FullRange>
        constexpr static float scale = FullRange ? fullMax : (Chroma ? (maxC - minC) : (maxY - minY));
        template<bool Chroma, bool FullRange>
        constexpr static float offset = Chroma ? zeroC : (FullRange ? fullMin : minY);
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        constexpr static float clampMin = (PreserveRangeOvershoot || FullRange) ? legalMin : (Chroma ? minC : minY);
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        constexpr static float clampMax = (PreserveRangeOvershoot || FullRange) ? legalMax : (Chroma ? maxC : maxY);
    public:
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        __host__ __device__ static inline uint8_t fromFloat(float value)
        {
            #ifdef  __CUDA_ARCH__
            return __float2uint_rn(
                clamp(
                    clampMin<Chroma, FullRange, PreserveRangeOvershoot>,
                    clampMax<Chroma, FullRange, PreserveRangeOvershoot>,
                    offset<Chroma, FullRange> + scale<Chroma, FullRange> * value
                )
            );
            #else
            return std::lround(
                clamp(
                    clampMin<Chroma, FullRange, PreserveRangeOvershoot>,
                    clampMax<Chroma, FullRange, PreserveRangeOvershoot>,
                    offset<Chroma, FullRange> + scale<Chroma, FullRange> * value
                )
            );
            #endif
        }
        template<bool Chroma, bool FullRange>
        __host__ __device__ static inline float toFloat(uint8_t value)
        {
            #ifdef  __CUDA_ARCH__
            return (__uint2float_rn(value) - offset<Chroma, FullRange>) / scale<Chroma, FullRange> * value;
            #else
            return (static_cast<float>(value) - offset<Chroma, FullRange>) / scale<Chroma, FullRange> * value;
            #endif
        }
    };
};

template<>
struct quantization<10>
{
    struct rgb
    {
    private:
        constexpr static float scale = 1023.0f;
    public:
        __host__ __device__ static inline uint16_t fromFloat(float value)
        {
            #ifdef  __CUDA_ARCH__
            return __float2uint_rn(value * scale);
            #else
            return std::lround(value * scale);
            #endif
        }
        __host__ __device__ static inline float toFloat(uint16_t value)
        {
            #ifdef  __CUDA_ARCH__
            return __uint2float_rn(value) / scale;
            #else
            return static_cast<float>(value) * scale;
            #endif
        }
    };

    struct yuv
    {
    private:
        constexpr static uint16_t minY = 64;
        constexpr static uint16_t maxY = 940;
        constexpr static uint16_t minC = 64;
        constexpr static uint16_t maxC = 960;
        constexpr static uint16_t fullMin = 0;
        constexpr static uint16_t fullMax = 1023;
        constexpr static uint16_t legalMin = 4;
        constexpr static uint16_t legalMax = 1019;
        constexpr static uint16_t zeroC = 512;

        template<bool Chroma, bool FullRange>
        constexpr static float scale = FullRange ? fullMax : (Chroma ? (maxC - minC) : (maxY - minY));
        template<bool Chroma, bool FullRange>
        constexpr static float offset = Chroma ? zeroC : (FullRange ? fullMin : minY);
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        constexpr static float clampMin = (PreserveRangeOvershoot || FullRange) ? legalMin : (Chroma ? minC : minY);
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        constexpr static float clampMax = (PreserveRangeOvershoot || FullRange) ? legalMax : (Chroma ? maxC : maxY);
    public:
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        __host__ __device__ static inline uint16_t fromFloat(float value)
        {
            #ifdef  __CUDA_ARCH__
            return __float2uint_rn(
                clamp(
                    clampMin<Chroma, FullRange, PreserveRangeOvershoot>,
                    clampMax<Chroma, FullRange, PreserveRangeOvershoot>,
                    offset<Chroma, FullRange> + scale<Chroma, FullRange> * value
                )
            );
            #else
            return std::lround(
                clamp(
                    clampMin<Chroma, FullRange, PreserveRangeOvershoot>,
                    clampMax<Chroma, FullRange, PreserveRangeOvershoot>,
                    offset<Chroma, FullRange> + scale<Chroma, FullRange> * value
                )
            );
            #endif
        }
        template<bool Chroma, bool FullRange>
        __host__ __device__ static inline float toFloat(uint16_t value)
        {
            #ifdef  __CUDA_ARCH__
            return (__uint2float_rn(value) - offset<Chroma, FullRange>) / scale<Chroma, FullRange> * value;
            #else
            return (static_cast<float>(value) - offset<Chroma, FullRange>) / scale<Chroma, FullRange> * value;
            #endif
        }
    };
};

template<>
struct quantization<16>
{
    struct rgb
    {
    private:
        constexpr static float scale = 65535.0f;
    public:
        __host__ __device__ static inline uint16_t fromFloat(float value)
        {
            #ifdef  __CUDA_ARCH__
            return __float2uint_rn(value * scale);
            #else
            return std::lround(value * scale);
            #endif
        }
        __host__ __device__ static inline float toFloat(uint16_t value)
        {
            #ifdef  __CUDA_ARCH__
            return __uint2float_rn(value) / scale;
            #else
            return static_cast<float>(value) * scale;
            #endif
        }
    };

    struct yuv
    {
    private:
        constexpr static uint16_t minY = 4096;
        constexpr static uint16_t maxY = 60160;
        constexpr static uint16_t minC = 4096;
        constexpr static uint16_t maxC = 61440;
        constexpr static uint16_t fullMin = 0;
        constexpr static uint16_t fullMax = 65535;
        constexpr static uint16_t legalMin = 256;
        constexpr static uint16_t legalMax = 65216;
        constexpr static uint16_t zeroC = 32768;

        template<bool Chroma, bool FullRange>
        constexpr static float scale = FullRange ? fullMax : (Chroma ? (maxC - minC) : (maxY - minY));
        template<bool Chroma, bool FullRange>
        constexpr static float offset = Chroma ? zeroC : (FullRange ? fullMin : minY);
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        constexpr static float clampMin = (PreserveRangeOvershoot || FullRange) ? legalMin : (Chroma ? minC : minY);
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        constexpr static float clampMax = (PreserveRangeOvershoot || FullRange) ? legalMax : (Chroma ? maxC : maxY);
    public:
        template<bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
        __host__ __device__ static inline uint16_t fromFloat(float value)
        {
            #ifdef  __CUDA_ARCH__
            return __float2uint_rn(
                clamp(
                    clampMin<Chroma, FullRange, PreserveRangeOvershoot>,
                    clampMax<Chroma, FullRange, PreserveRangeOvershoot>,
                    offset<Chroma, FullRange> + scale<Chroma, FullRange> * value
                )
            );
            #else
            return std::lround(
                clamp(
                    clampMin<Chroma, FullRange, PreserveRangeOvershoot>,
                    clampMax<Chroma, FullRange, PreserveRangeOvershoot>,
                    offset<Chroma, FullRange> + scale<Chroma, FullRange> * value
                )
            );
            #endif
        }
        template<bool Chroma, bool FullRange>
        __host__ __device__ static inline float toFloat(uint16_t value)
        {
            #ifdef  __CUDA_ARCH__
            return (__uint2float_rn(value) - offset<Chroma, FullRange>) / scale<Chroma, FullRange> * value;
            #else
            return (static_cast<float>(value) - offset<Chroma, FullRange>) / scale<Chroma, FullRange> * value;
            #endif
        }
    };
};

template<size_t Bits>
__host__ __device__ static inline auto fromRGBFloat(float value)
{
    return quantization<Bits>::rgb::fromFloat(value);
}

template<size_t Bits, typename Type>
__host__ __device__ static inline float toRGBFloat(Type value)
{
    return quantization<Bits>::rgb::toFloat(value);
}

template<size_t Bits, bool Chroma, bool FullRange, bool PreserveRangeOvershoot>
__host__ __device__ static inline auto fromYUVFloat(float value)
{
    return quantization<Bits>::yuv::fromFloat<Chroma, FullRange, PreserveRangeOvershoot>(value);
}

template<size_t Bits, bool Chroma>
__host__ __device__ static inline auto fromYUVFloat(float value, bool fullRange, bool preserveRangeOvershoot)
{
    uint8_t flags = (fullRange * 0b10) | (preserveRangeOvershoot * 0b01);
    switch (flags)
    {
        default:
        case 0b00:
            return quantization<Bits>::yuv::fromFloat<Chroma, false, false>(value);
        case 0b01:
            return quantization<Bits>::yuv::fromFloat<Chroma, false, true>(value);
        case 0b10:
            return quantization<Bits>::yuv::fromFloat<Chroma, true, false>(value);
        case 0b11:
            return quantization<Bits>::yuv::fromFloat<Chroma, true, true>(value);
    }
}

template<size_t Bits, bool Chroma, typename Type>
__host__ __device__ static inline float toYUVFloat(Type value, bool fullRange)
{
    uint8_t flags = (fullRange * 0b1);
    switch (flags)
    {
        default:
        case 0b0:
            return quantization<Bits>::yuv::toFloat<Chroma, false>(value);
        case 0b1:
            return quantization<Bits>::yuv::toFloat<Chroma, true>(value);
    }
}