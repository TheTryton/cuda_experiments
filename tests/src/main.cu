#include <8bit.cuh>
#include <10bit.cuh>
#include <16bit.cuh>

#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
#include <tuple>

void yuv444p(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
             color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv444PPlaneDeviceData[3];
    size_t yuv444PPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv444PPlaneDeviceData[0], &yuv444PPlaneDeviceDataPitch[0], width * sizeof(yuv8bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv444PPlaneDeviceData[1], &yuv444PPlaneDeviceDataPitch[1], width * sizeof(yuv8bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv444PPlaneDeviceData[2], &yuv444PPlaneDeviceDataPitch[2], width * sizeof(yuv8bit_t), height));

    dim3 blockSizeYUV444P = {
        32, 32, 1
    };
    dim3 gridSizeYUV444P = {
        (width/yuv444pBlockSize + blockSizeYUV444P.x - 1) / blockSizeYUV444P.x,
        (height + blockSizeYUV444P.y - 1) / blockSizeYUV444P.y,
        1
    };
    kernel_yuv444p_output_single_pass<<<gridSizeYUV444P, blockSizeYUV444P, 0, stream>>>(
        rgbaTexture,
        yuv444PPlaneDeviceData[0], yuv444PPlaneDeviceData[1], yuv444PPlaneDeviceData[2],
        yuv444PPlaneDeviceDataPitch[0], yuv444PPlaneDeviceDataPitch[1], yuv444PPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv444PPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv444PPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv444PPlaneDeviceData[2]));
}

void yuv422p(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
             color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv422PPlaneDeviceData[3];
    size_t yuv422PPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv422PPlaneDeviceData[0], &yuv422PPlaneDeviceDataPitch[0], width * sizeof(yuv8bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv422PPlaneDeviceData[1], &yuv422PPlaneDeviceDataPitch[1], width / 2 * sizeof(yuv8bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv422PPlaneDeviceData[2], &yuv422PPlaneDeviceDataPitch[2], width / 2 * sizeof(yuv8bit_t), height));

    dim3 blockSizeYUV422P = {
        32, 32, 1
    };
    dim3 gridSizeYUV422P = {
        (width/yuv422pBlockSize + blockSizeYUV422P.x - 1) / blockSizeYUV422P.x,
        (height + blockSizeYUV422P.y - 1) / blockSizeYUV422P.y,
        1
    };
    kernel_yuv422p_output_single_pass<<<gridSizeYUV422P, blockSizeYUV422P, 0, stream>>>(
        rgbaTexture,
        yuv422PPlaneDeviceData[0], yuv422PPlaneDeviceData[1], yuv422PPlaneDeviceData[2],
        yuv422PPlaneDeviceDataPitch[0], yuv422PPlaneDeviceDataPitch[1], yuv422PPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv422PPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv422PPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv422PPlaneDeviceData[2]));
}

void yuv420p(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
             color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv420PPlaneDeviceData[3];
    size_t yuv420PPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv420PPlaneDeviceData[0], &yuv420PPlaneDeviceDataPitch[0], width * sizeof(yuv8bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv420PPlaneDeviceData[1], &yuv420PPlaneDeviceDataPitch[1], width / 2 * sizeof(yuv8bit_t), height / 2));
    cudaAssert(cudaMallocPitch(&yuv420PPlaneDeviceData[2], &yuv420PPlaneDeviceDataPitch[2], width / 2 * sizeof(yuv8bit_t), height / 2));

    dim3 blockSizeYUV420P = {
        32, 32, 1
    };
    dim3 gridSizeYUV420P = {
        (width/yuv420pBlockSize + blockSizeYUV420P.x - 1) / blockSizeYUV420P.x,
        (height + blockSizeYUV420P.y - 1) / blockSizeYUV420P.y,
        1
    };
    kernel_yuv420p_output_single_pass<<<gridSizeYUV420P, blockSizeYUV420P, 0, stream>>>(
        rgbaTexture,
        yuv420PPlaneDeviceData[0], yuv420PPlaneDeviceData[1], yuv420PPlaneDeviceData[2],
        yuv420PPlaneDeviceDataPitch[0], yuv420PPlaneDeviceDataPitch[1], yuv420PPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv420PPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv420PPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv420PPlaneDeviceData[2]));
}

void yuv444p10le(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
                 color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv444P10LEPlaneDeviceData[3];
    size_t yuv444P10LEPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv444P10LEPlaneDeviceData[0], &yuv444P10LEPlaneDeviceDataPitch[0], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv444P10LEPlaneDeviceData[1], &yuv444P10LEPlaneDeviceDataPitch[1], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv444P10LEPlaneDeviceData[2], &yuv444P10LEPlaneDeviceDataPitch[2], width * sizeof(yuv10bit_t), height));

    dim3 blockSizeYUV444P10LE = {
        32, 32, 1
    };
    dim3 gridSizeYUV444P10LE = {
        (width/yuv444p10leBlockSize + blockSizeYUV444P10LE.x - 1) / blockSizeYUV444P10LE.x,
        (height + blockSizeYUV444P10LE.y - 1) / blockSizeYUV444P10LE.y,
        1
    };
    kernel_yuv444p10le_output_single_pass<<<gridSizeYUV444P10LE, blockSizeYUV444P10LE, 0, stream>>>(
        rgbaTexture,
        yuv444P10LEPlaneDeviceData[0], yuv444P10LEPlaneDeviceData[1], yuv444P10LEPlaneDeviceData[2],
        yuv444P10LEPlaneDeviceDataPitch[0], yuv444P10LEPlaneDeviceDataPitch[1], yuv444P10LEPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv444P10LEPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv444P10LEPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv444P10LEPlaneDeviceData[2]));
}

void yuv422p10le(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
                 color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv422P10LEPlaneDeviceData[3];
    size_t yuv422P10LEPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv422P10LEPlaneDeviceData[0], &yuv422P10LEPlaneDeviceDataPitch[0], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv422P10LEPlaneDeviceData[1], &yuv422P10LEPlaneDeviceDataPitch[1], width / 2 * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv422P10LEPlaneDeviceData[2], &yuv422P10LEPlaneDeviceDataPitch[2], width / 2 * sizeof(yuv10bit_t), height));

    dim3 blockSizeYUV422P10LE = {
        32, 32, 1
    };
    dim3 gridSizeYUV422P10LE = {
        (width/yuv422p10leBlockSize + blockSizeYUV422P10LE.x - 1) / blockSizeYUV422P10LE.x,
        (height + blockSizeYUV422P10LE.y - 1) / blockSizeYUV422P10LE.y,
        1
    };
    kernel_yuv422p10le_output_single_pass<<<gridSizeYUV422P10LE, blockSizeYUV422P10LE, 0, stream>>>(
        rgbaTexture,
        yuv422P10LEPlaneDeviceData[0], yuv422P10LEPlaneDeviceData[1], yuv422P10LEPlaneDeviceData[2],
        yuv422P10LEPlaneDeviceDataPitch[0], yuv422P10LEPlaneDeviceDataPitch[1], yuv422P10LEPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv422P10LEPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv422P10LEPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv422P10LEPlaneDeviceData[2]));
}

void yuv420p10le(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
                 color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv420P10LEPlaneDeviceData[3];
    size_t yuv420P10LEPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv420P10LEPlaneDeviceData[0], &yuv420P10LEPlaneDeviceDataPitch[0], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv420P10LEPlaneDeviceData[1], &yuv420P10LEPlaneDeviceDataPitch[1], width / 2 * sizeof(yuv10bit_t), height / 2));
    cudaAssert(cudaMallocPitch(&yuv420P10LEPlaneDeviceData[2], &yuv420P10LEPlaneDeviceDataPitch[2], width / 2 * sizeof(yuv10bit_t), height / 2));

    dim3 blockSizeYUV420P10LE = {
        32, 32, 1
    };
    dim3 gridSizeYUV420P10LE = {
        (width/yuv420p10leBlockSize + blockSizeYUV420P10LE.x - 1) / blockSizeYUV420P10LE.x,
        (height + blockSizeYUV420P10LE.y - 1) / blockSizeYUV420P10LE.y,
        1
    };
    kernel_yuv420p10le_output_single_pass<<<gridSizeYUV420P10LE, blockSizeYUV420P10LE, 0, stream>>>(
        rgbaTexture,
        yuv420P10LEPlaneDeviceData[0], yuv420P10LEPlaneDeviceData[1], yuv420P10LEPlaneDeviceData[2],
        yuv420P10LEPlaneDeviceDataPitch[0], yuv420P10LEPlaneDeviceDataPitch[1], yuv420P10LEPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv420P10LEPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv420P10LEPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv420P10LEPlaneDeviceData[2]));
}

void yuv444p16le(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
                 color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv444P16LEPlaneDeviceData[3];
    size_t yuv444P16LEPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv444P16LEPlaneDeviceData[0], &yuv444P16LEPlaneDeviceDataPitch[0], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv444P16LEPlaneDeviceData[1], &yuv444P16LEPlaneDeviceDataPitch[1], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv444P16LEPlaneDeviceData[2], &yuv444P16LEPlaneDeviceDataPitch[2], width * sizeof(yuv10bit_t), height));

    dim3 blockSizeYUV444P16LE = {
        32, 32, 1
    };
    dim3 gridSizeYUV444P16LE = {
        (width/yuv444p16leBlockSize + blockSizeYUV444P16LE.x - 1) / blockSizeYUV444P16LE.x,
        (height + blockSizeYUV444P16LE.y - 1) / blockSizeYUV444P16LE.y,
        1
    };
    kernel_yuv444p16le_output_single_pass<<<gridSizeYUV444P16LE, blockSizeYUV444P16LE, 0, stream>>>(
        rgbaTexture,
        yuv444P16LEPlaneDeviceData[0], yuv444P16LEPlaneDeviceData[1], yuv444P16LEPlaneDeviceData[2],
        yuv444P16LEPlaneDeviceDataPitch[0], yuv444P16LEPlaneDeviceDataPitch[1], yuv444P16LEPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv444P16LEPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv444P16LEPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv444P16LEPlaneDeviceData[2]));
}

void yuv422p16le(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
                 color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv422P16LEPlaneDeviceData[3];
    size_t yuv422P16LEPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv422P16LEPlaneDeviceData[0], &yuv422P16LEPlaneDeviceDataPitch[0], width * sizeof(yuv16bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv422P16LEPlaneDeviceData[1], &yuv422P16LEPlaneDeviceDataPitch[1], width / 2 * sizeof(yuv16bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv422P16LEPlaneDeviceData[2], &yuv422P16LEPlaneDeviceDataPitch[2], width / 2 * sizeof(yuv16bit_t), height));

    dim3 blockSizeYUV422P16LE = {
        32, 32, 1
    };
    dim3 gridSizeYUV422P16LE = {
        (width/yuv422p16leBlockSize + blockSizeYUV422P16LE.x - 1) / blockSizeYUV422P16LE.x,
        (height + blockSizeYUV422P16LE.y - 1) / blockSizeYUV422P16LE.y,
        1
    };
    kernel_yuv422p16le_output_single_pass<<<gridSizeYUV422P16LE, blockSizeYUV422P16LE, 6, stream>>>(
        rgbaTexture,
        yuv422P16LEPlaneDeviceData[0], yuv422P16LEPlaneDeviceData[1], yuv422P16LEPlaneDeviceData[2],
        yuv422P16LEPlaneDeviceDataPitch[0], yuv422P16LEPlaneDeviceDataPitch[1], yuv422P16LEPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv422P16LEPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv422P16LEPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv422P16LEPlaneDeviceData[2]));
}

void yuv420p16le(cudaTextureObject_t rgbaTexture, uint32_t width, uint32_t height,
                 color_space colorSpace, bool fullRange, bool preserveRangeOvershoot, cudaStream_t stream)
{
    uint8_t* yuv420P16LEPlaneDeviceData[3];
    size_t yuv420P16LEPlaneDeviceDataPitch[3];
    cudaAssert(cudaMallocPitch(&yuv420P16LEPlaneDeviceData[0], &yuv420P16LEPlaneDeviceDataPitch[0], width * sizeof(yuv10bit_t), height));
    cudaAssert(cudaMallocPitch(&yuv420P16LEPlaneDeviceData[1], &yuv420P16LEPlaneDeviceDataPitch[1], width / 2 * sizeof(yuv10bit_t), height / 2));
    cudaAssert(cudaMallocPitch(&yuv420P16LEPlaneDeviceData[2], &yuv420P16LEPlaneDeviceDataPitch[2], width / 2 * sizeof(yuv10bit_t), height / 2));

    dim3 blockSizeYUV420P16LE = {
        32, 32, 1
    };
    dim3 gridSizeYUV420P16LE = {
        (width/yuv420p16leBlockSize + blockSizeYUV420P16LE.x - 1) / blockSizeYUV420P16LE.x,
        (height + blockSizeYUV420P16LE.y - 1) / blockSizeYUV420P16LE.y,
        1
    };
    kernel_yuv420p16le_output_single_pass<<<gridSizeYUV420P16LE, blockSizeYUV420P16LE, 0, stream>>>(
        rgbaTexture,
        yuv420P16LEPlaneDeviceData[0], yuv420P16LEPlaneDeviceData[1], yuv420P16LEPlaneDeviceData[2],
        yuv420P16LEPlaneDeviceDataPitch[0], yuv420P16LEPlaneDeviceDataPitch[1], yuv420P16LEPlaneDeviceDataPitch[2],
        width, height,
        colorSpace, fullRange, preserveRangeOvershoot
    );

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaFree(yuv420P16LEPlaneDeviceData[0]));
    cudaAssert(cudaFree(yuv420P16LEPlaneDeviceData[1]));
    cudaAssert(cudaFree(yuv420P16LEPlaneDeviceData[2]));
};

std::tuple<uint8_t*, cudaTextureObject_t> createR8G8B8A8Image(uint32_t width, uint32_t height)
{
    using imagePixelType_t = uchar4;
    const size_t rgbaImageDataSize = width * height * sizeof(imagePixelType_t);
    imagePixelType_t * rgbaHostImageData;
    cudaAssert(cudaMallocHost(&rgbaHostImageData, rgbaImageDataSize));
    std::generate_n(rgbaHostImageData, width * height, [](){
        static std::random_device rd;
        static std::mt19937 generator(rd());
        static std::uniform_int_distribution<uint16_t> distribution(0, std::numeric_limits<uint8_t>::max());
        return imagePixelType_t{
            static_cast<uint8_t>(distribution(generator)),
            static_cast<uint8_t>(distribution(generator)),
            static_cast<uint8_t>(distribution(generator)),
            static_cast<uint8_t>(distribution(generator))
        };
    });

    uint8_t * rgbaDeviceImageData;
    size_t rgbaDeviceImageDataPitch;
    cudaAssert(cudaMallocPitch(
        &rgbaDeviceImageData, &rgbaDeviceImageDataPitch,
        width * sizeof(imagePixelType_t), height)
    );

    cudaResourceDesc resourceDesc{};
    resourceDesc.resType = cudaResourceType::cudaResourceTypePitch2D;
    resourceDesc.res.pitch2D.width = width;
    resourceDesc.res.pitch2D.height = height;
    resourceDesc.res.pitch2D.pitchInBytes = rgbaDeviceImageDataPitch;
    resourceDesc.res.pitch2D.devPtr = rgbaDeviceImageData;
    resourceDesc.res.pitch2D.desc = cudaCreateChannelDesc<imagePixelType_t>();

    cudaTextureDesc textureDesc{};
    textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
    textureDesc.normalizedCoords = false;
    textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
    textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;

    cudaTextureObject_t rgbaTexture;
    cudaAssert(cudaCreateTextureObject(&rgbaTexture, &resourceDesc, &textureDesc, nullptr));

    cudaAssert(cudaMemcpy2D(
        rgbaDeviceImageData, rgbaDeviceImageDataPitch,
        rgbaHostImageData, width * sizeof(imagePixelType_t),
        width, height, cudaMemcpyKind::cudaMemcpyHostToDevice)
    );

    cudaAssert(cudaFreeHost(rgbaHostImageData));

    return std::make_tuple(rgbaDeviceImageData, rgbaTexture);
}

std::tuple<uint8_t*, cudaTextureObject_t> createR16G16B16A16Image(uint32_t width, uint32_t height)
{
    using imagePixelType_t = ushort4;
    const size_t rgbaImageDataSize = width * height * sizeof(imagePixelType_t);
    imagePixelType_t * rgbaHostImageData;
    cudaAssert(cudaMallocHost(&rgbaHostImageData, rgbaImageDataSize));
    std::generate_n(rgbaHostImageData, width * height, [](){
        static std::random_device rd;
        static std::mt19937 generator(rd());
        static std::uniform_int_distribution<uint16_t> distribution(0, std::numeric_limits<uint16_t>::max());
        return imagePixelType_t{
            static_cast<uint16_t>(distribution(generator)),
            static_cast<uint16_t>(distribution(generator)),
            static_cast<uint16_t>(distribution(generator)),
            static_cast<uint16_t>(distribution(generator))
        };
    });

    uint8_t * rgbaDeviceImageData;
    size_t rgbaDeviceImageDataPitch;
    cudaAssert(cudaMallocPitch(
        &rgbaDeviceImageData, &rgbaDeviceImageDataPitch,
        width * sizeof(imagePixelType_t), height)
    );

    cudaResourceDesc resourceDesc{};
    resourceDesc.resType = cudaResourceType::cudaResourceTypePitch2D;
    resourceDesc.res.pitch2D.width = width;
    resourceDesc.res.pitch2D.height = height;
    resourceDesc.res.pitch2D.pitchInBytes = rgbaDeviceImageDataPitch;
    resourceDesc.res.pitch2D.devPtr = rgbaDeviceImageData;
    resourceDesc.res.pitch2D.desc = cudaCreateChannelDesc<imagePixelType_t>();

    cudaTextureDesc textureDesc{};
    textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
    textureDesc.normalizedCoords = false;
    textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
    textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;

    cudaTextureObject_t rgbaTexture;
    cudaAssert(cudaCreateTextureObject(&rgbaTexture, &resourceDesc, &textureDesc, nullptr));

    cudaAssert(cudaMemcpy2D(
        rgbaDeviceImageData, rgbaDeviceImageDataPitch,
        rgbaHostImageData, width * sizeof(imagePixelType_t),
        width, height, cudaMemcpyKind::cudaMemcpyHostToDevice)
    );

    cudaAssert(cudaFreeHost(rgbaHostImageData));

    return std::make_tuple(rgbaDeviceImageData, rgbaTexture);
}

std::tuple<uint8_t*, cudaTextureObject_t> createR32G32B32A32Image(uint32_t width, uint32_t height)
{
    using imagePixelType_t = uint4;
    const size_t rgbaImageDataSize = width * height * sizeof(imagePixelType_t);
    imagePixelType_t * rgbaHostImageData;
    cudaAssert(cudaMallocHost(&rgbaHostImageData, rgbaImageDataSize));
    std::generate_n(rgbaHostImageData, width * height, [](){
        static std::random_device rd;
        static std::mt19937 generator(rd());
        static std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
        return imagePixelType_t{
            static_cast<uint32_t>(distribution(generator)),
            static_cast<uint32_t>(distribution(generator)),
            static_cast<uint32_t>(distribution(generator)),
            static_cast<uint32_t>(distribution(generator))
        };
    });

    uint8_t * rgbaDeviceImageData;
    size_t rgbaDeviceImageDataPitch;
    cudaAssert(cudaMallocPitch(
        &rgbaDeviceImageData, &rgbaDeviceImageDataPitch,
        width * sizeof(imagePixelType_t), height)
    );

    cudaResourceDesc resourceDesc{};
    resourceDesc.resType = cudaResourceType::cudaResourceTypePitch2D;
    resourceDesc.res.pitch2D.width = width;
    resourceDesc.res.pitch2D.height = height;
    resourceDesc.res.pitch2D.pitchInBytes = rgbaDeviceImageDataPitch;
    resourceDesc.res.pitch2D.devPtr = rgbaDeviceImageData;
    resourceDesc.res.pitch2D.desc = cudaCreateChannelDesc<imagePixelType_t>();

    cudaTextureDesc textureDesc{};
    textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
    textureDesc.normalizedCoords = false;
    textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
    textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;

    cudaTextureObject_t rgbaTexture;
    cudaAssert(cudaCreateTextureObject(&rgbaTexture, &resourceDesc, &textureDesc, nullptr));

    cudaAssert(cudaMemcpy2D(
        rgbaDeviceImageData, rgbaDeviceImageDataPitch,
        rgbaHostImageData, width * sizeof(imagePixelType_t),
        width, height, cudaMemcpyKind::cudaMemcpyHostToDevice)
    );

    cudaAssert(cudaFreeHost(rgbaHostImageData));

    return std::make_tuple(rgbaDeviceImageData, rgbaTexture);
}

__global__ void shortKernel(float * outDevice, float * inDevice, size_t countElements){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<countElements) outDevice[idx]=1.23*inDevice[idx];
}

void checkGraphPerformance(uint32_t countElements, cudaStream_t stream)
{
    using namespace std;

    constexpr size_t kernelInvocations = 200;
    constexpr size_t countRuns = 1000;

    float *outDevice;
    cudaAssert(cudaMalloc(&outDevice, countElements * sizeof(float)));
    float *inDevice;
    cudaAssert(cudaMalloc(&inDevice, countElements * sizeof(float)));

    cudaEvent_t start, stop;
    cudaAssert(cudaEventCreate(&start));
    cudaAssert(cudaEventCreate(&stop));

    const uint32_t blockDim = 128;
    const uint32_t gridDim = (countElements + blockDim - 1) / blockDim;

    cudaAssert(cudaEventRecord(start, stream));
    for(int istep=0; istep<countRuns; istep++)
    {
        for(int ikrnl=0; ikrnl<kernelInvocations; ikrnl++)
        {
            shortKernel<<<gridDim, blockDim, 0, stream>>>(outDevice, inDevice);
        }
        cudaAssert(cudaStreamSynchronize(stream));
    }
    cudaAssert(cudaEventRecord(stop, stream));

    cudaAssert(cudaStreamSynchronize(stream));

    float milliseconds;
    cudaAssert(cudaEventElapsedTime(&milliseconds, start, stop));

    cout << "GPU Took (without graph): " << milliseconds << "ms" << std::endl;

    cudaGraph_t graph;
    cudaGraphExec_t graphInstance;
    cudaAssert(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for(int ikrnl=0; ikrnl<kernelInvocations; ikrnl++){
        shortKernel<<<gridDim, blockDim, 0, stream>>>(outDevice, inDevice);
    }
    cudaAssert(cudaStreamEndCapture(stream, &graph));
    cudaAssert(cudaGraphInstantiate(&graphInstance, graph, nullptr, nullptr, 0));

    cudaAssert(cudaEventRecord(start, stream));
    for(int istep=0; istep<countRuns; istep++){
        cudaAssert(cudaGraphLaunch(graphInstance, stream));
    }
    cudaAssert(cudaEventRecord(stop, stream));

    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaEventElapsedTime(&milliseconds, start, stop));

    cout << "GPU Took (with graph): " << milliseconds << "ms" << std::endl;

    cudaAssert(cudaEventDestroy(stop));
    cudaAssert(cudaEventDestroy(start));
}

int main(int argc, const char* argv[])
{
    constexpr uint32_t width = 3840 * 4;
    constexpr uint32_t height = 2160  * 4;
    constexpr color_space colorSpace = color_space::rec2020;
    constexpr bool fullRange = true;
    constexpr bool preserveRangeOvershoot = true;
    constexpr uint32_t countElements = 65536;

    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));

    // checkGraphPerformance(countElements, stream);

    // R8G8B8A8
    {
        auto [r8g8b8a8DeviceImageData, r8g8b8a8Texture] = createR8G8B8A8Image(width, height);

        yuv444p(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        yuv444p10le(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p10le(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p10le(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        yuv444p16le(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p16le(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p16le(r8g8b8a8Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        cudaAssert(cudaDestroyTextureObject(r8g8b8a8Texture));
        cudaAssert(cudaFree(r8g8b8a8DeviceImageData));
    }

    // R16G16B16A16
    {
        auto [r16g16b16a16DeviceImageData, r16g16b16a16Texture] = createR8G8B8A8Image(width, height);

        yuv444p(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        yuv444p10le(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p10le(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p10le(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        yuv444p16le(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p16le(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p16le(r16g16b16a16Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        cudaAssert(cudaDestroyTextureObject(r16g16b16a16Texture));
        cudaAssert(cudaFree(r16g16b16a16DeviceImageData));
    }

    // R32G32B32A32
    {
        auto [r32g32b32a32DeviceImageData, r32g32b32a32Texture] = createR8G8B8A8Image(width, height);

        yuv444p(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        yuv444p10le(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p10le(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p10le(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        yuv444p16le(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv422p16le(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);
        yuv420p16le(r32g32b32a32Texture, width, height, colorSpace, fullRange, preserveRangeOvershoot, stream);

        cudaAssert(cudaDestroyTextureObject(r32g32b32a32Texture));
        cudaAssert(cudaFree(r32g32b32a32DeviceImageData));
    }

    cudaAssert(cudaStreamSynchronize(stream));
    cudaAssert(cudaStreamDestroy(stream));

    return 0;
}

/*#include <iostream>
#include <cassert>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

void cudaAssert(cudaError_t error)
{
    if(error != cudaSuccess)
    {
        assert(error == cudaSuccess);
    }
}

class stream_t
{
private:
    constexpr static size_t s_maxStreamQueuedOperationsCount = 16;
private:
    cudaStream_t m_stream{};
    mutable std::mutex m_mutex;
    std::condition_variable m_conditionVariable;
    size_t m_streamScheduledOperationsCount{};
    std::queue<std::function<void()>> m_operationsWaitingToBeQueued;

    size_t m_operationIndex{};
    size_t m_finishedOperationIndex{};
public:
    stream_t()
    {
        cudaAssert(cudaStreamCreate(&m_stream));
    }
    stream_t(const stream_t& other) = delete;
    stream_t(stream_t&& other) = delete;
public:
    stream_t& operator=(const stream_t& other) = delete;
    stream_t& operator=(stream_t&& other) = delete;
public:
    ~stream_t()
    {
        waitFinish();
        cudaAssert(cudaStreamSynchronize(m_stream));
        cudaAssert(cudaStreamDestroy(m_stream));
        m_stream = {};
    }
private:
    static void cudaOperationFinished(cudaStream_t _stream, cudaError_t _status, void *userData)
    {
        auto& stream = *static_cast<stream_t*>(userData);
        {
            std::unique_lock lock{stream.m_mutex};
            --stream.m_streamScheduledOperationsCount;
            std::cout << "Finished operation " << stream.m_finishedOperationIndex++ << std::endl;
        }

        stream.m_conditionVariable.notify_one();
    }
private:
    template<typename Operation>
    void scheduleCudaOperation(Operation&& operation)
    {
        std::unique_lock lock{m_mutex};

        if (m_streamScheduledOperationsCount < s_maxStreamQueuedOperationsCount)
        {
            std::cout << "Scheduling operation " << m_operationIndex++ << std::endl;
            ++m_streamScheduledOperationsCount;
            lock.unlock();
            operation();
        }
        else
        {
            std::cout << "Operation " << m_operationIndex++ << " pushed to wait queue" << std::endl;
            m_operationsWaitingToBeQueued.emplace(std::forward<Operation>(operation));
        }
    }
public:
    template<typename Operation, typename... Arguments>
    void scheduleCudaOperation(Operation&& cudaStreamOperation, Arguments... arguments)
    {
        auto operation = [cudaStreamOperation = std::forward<Operation>(cudaStreamOperation), arguments..., this]() {
            cudaAssert(cudaStreamOperation(arguments..., m_stream));
            cudaAssert(cudaStreamAddCallback(m_stream, cudaOperationFinished, this, 0));
        };

        scheduleCudaOperation(std::move(operation));
    }

    void waitFinish()
    {
        std::unique_lock lock(m_mutex);
        while (!m_operationsWaitingToBeQueued.empty())
        {
            m_conditionVariable.wait(lock, [this]() {
                return m_streamScheduledOperationsCount < s_maxStreamQueuedOperationsCount;
            });
            auto operation = std::move(m_operationsWaitingToBeQueued.front());
            m_operationsWaitingToBeQueued.pop();
            ++m_streamScheduledOperationsCount;
            std::cout << "Scheduling operation from wait queue" << std::endl;
            lock.unlock();
            operation();
            lock.lock();
        }
    }
};

__global__ void kernel0(const uint32_t* input, uint32_t* output, size_t size)
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index >= size)
        return;

    output[index] = input[index];
    //for(int i=0;i<size;i+=size/8)
    //    output[index] += input[i];
}

int main() {
    constexpr size_t dataCount = 1024ull * 1024 * 512;
    constexpr size_t dataSize = dataCount * sizeof(uint32_t);

    uint32_t* inputHost;
    cudaAssert(cudaMallocHost(&inputHost, dataSize));
    uint32_t* outputHost;
    cudaAssert(cudaMallocHost(&outputHost, dataSize));
    uint32_t* inputDevice;
    cudaAssert(cudaMalloc(&inputDevice, dataSize));
    uint32_t* outputDevice;
    cudaAssert(cudaMalloc(&outputDevice, dataSize));

    size_t dc = dataCount;
    size_t ds = dataSize;
    void* args[] = {
        &inputDevice,
        &outputDevice,
        &dc
    };

    const auto scheduleWork = [&](stream_t& stream){
        stream.scheduleCudaOperation(
            cudaMemcpyAsync,
            static_cast<void*>(inputDevice),
            static_cast<const void*>(inputHost),
            ds,
            cudaMemcpyKind::cudaMemcpyHostToDevice
        );

        uint32_t gridDim = (dataSize + 512 - 1)/ 512;

        stream.scheduleCudaOperation(
            &cudaLaunchKernel<void>,
            static_cast<const void*>(kernel0),
            dim3{gridDim, 1, 1},
            dim3{512,1,1},
            args,
            static_cast<size_t>(0)
        );

        stream.scheduleCudaOperation(
            cudaMemcpyAsync,
            static_cast<void*>(outputHost),
            static_cast<const void*>(outputDevice),
            ds,
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        );
    };

    {
        stream_t stream;

        constexpr size_t invocations = 64;

        for(size_t i = 0; i < invocations; i++)
        {
            scheduleWork(stream);
        }

        stream.waitFinish();
    }

    cudaAssert(cudaFree(inputDevice));
    cudaAssert(cudaFree(outputDevice));
    cudaAssert(cudaFreeHost(inputHost));
    cudaAssert(cudaFreeHost(outputHost));
    return 0;
}
*/