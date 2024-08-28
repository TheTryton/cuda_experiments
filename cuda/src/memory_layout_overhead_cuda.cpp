#include <memory_layout_overhead_cuda.hpp>

#include <cuda_helpers.hpp>

namespace memory_layout_overhead
{

template<size_t xBatchSize>
class variant2D_cuda_impl_non_pitched_intermediate_memory
    : public variant2D_impl
{
private:
    CUfunction _kernel;
    CUdeviceptr _textureObjectBuffer;
    CUtexObject _textureObject;
    CUdeviceptr _outputBuffer;
    CUevent _eventStart;
    CUevent _eventStop;
    size_t _width;
    size_t _height;
    size_t _blockSizeX;
    size_t _blockSizeY;
public:
    variant2D_cuda_impl_non_pitched_intermediate_memory(
        CUfunction kernel,
        size_t width,
        size_t height,
        std::span<const float> in,
        size_t inPitchInBytes,
        size_t blockSizeX,
        size_t blockSizeY
    )
        : _kernel(kernel)
          , _width(width)
          , _height(height)
          , _blockSizeX(blockSizeX)
          , _blockSizeY(blockSizeY)
    {
        if (_height * inPitchInBytes > in.size_bytes())
            throw std::exception("height * inPitchInBytes > in.size_bytes()!");

        check_error(cuMemAlloc(&_outputBuffer, 4 * sizeof(uint16_t) * width * height));

        size_t pitchInBytesTexture;
        check_error(cuMemAllocPitch(&_textureObjectBuffer, &pitchInBytesTexture, 4 * sizeof(float) * width, height, 4 * sizeof(float)));
        CUDA_RESOURCE_DESC texResDesc{};
        texResDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
        texResDesc.res.pitch2D.devPtr = _textureObjectBuffer;
        texResDesc.res.pitch2D.width = _width;
        texResDesc.res.pitch2D.height = _height;
        texResDesc.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
        texResDesc.res.pitch2D.numChannels = 4;
        texResDesc.res.pitch2D.pitchInBytes = pitchInBytesTexture;
        CUDA_TEXTURE_DESC texDesc{};
        texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        check_error(cuTexObjectCreate(&_textureObject, &texResDesc, &texDesc, nullptr));

        CUDA_MEMCPY2D memcpy2D{};
        memcpy2D.srcMemoryType = CU_MEMORYTYPE_HOST;
        memcpy2D.srcHost = in.data();
        memcpy2D.srcPitch = inPitchInBytes;
        memcpy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy2D.dstDevice = _textureObjectBuffer;
        memcpy2D.dstPitch = pitchInBytesTexture;
        memcpy2D.WidthInBytes = 4 * sizeof(float) * _width;
        memcpy2D.Height = _height;
        check_error(cuMemcpy2D(&memcpy2D));

        check_error(cuEventCreate(&_eventStart, 0));
        check_error(cuEventCreate(&_eventStop, 0));
    }
public:
    ~variant2D_cuda_impl_non_pitched_intermediate_memory() override
    {
        cuEventDestroy(_eventStart);
        cuEventDestroy(_eventStop);

        cuMemFree(_outputBuffer);
        cuTexObjectDestroy(_textureObject);
        cuMemFree(_textureObjectBuffer);
    }
public:
    variant_result execute(std::span<uint16_t> out, size_t outPitchInBytes) override
    {
        if (_height * outPitchInBytes > out.size_bytes())
            throw std::exception("height * outPitchInBytes > out.size_bytes()!");

        size_t gridSizeX = (_width + _blockSizeX * xBatchSize - 1) / (_blockSizeX * xBatchSize);
        size_t gridSizeY = (_height + _blockSizeY - 1) / _blockSizeY;
        void *args[] = {
            &_width,
            &_height,
            &_textureObject,
            &_outputBuffer
        };
        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuLaunchKernel(
            _kernel,
            gridSizeX, gridSizeY, 1,
            _blockSizeX, _blockSizeY, 1,
            0,
            nullptr,
            args,
            nullptr
        ));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float executeMilliseconds;
        check_error(cuEventElapsedTime(&executeMilliseconds, _eventStart, _eventStop));

        CUDA_MEMCPY2D memcpy2D{};
        memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy2D.srcDevice = _outputBuffer;
        memcpy2D.dstMemoryType = CU_MEMORYTYPE_HOST;
        memcpy2D.dstHost = out.data();
        memcpy2D.dstPitch = outPitchInBytes;
        memcpy2D.WidthInBytes = 4 * sizeof(uint16_t) * _width;
        memcpy2D.Height = _height;
        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpy2D(&memcpy2D));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float downloadMilliseconds;
        check_error(cuEventElapsedTime(&downloadMilliseconds, _eventStart, _eventStop));

        return {
            ._execute = std::chrono::nanoseconds{static_cast<long long>(executeMilliseconds * 1e6f)},
            ._download = std::chrono::nanoseconds{static_cast<long long>(downloadMilliseconds * 1e6f)},
        };
    }
};

template<size_t xBatchSize>
class variant2D_cuda_impl_pitched_intermediate_memory
    : public variant2D_impl
{
private:
    CUfunction _kernel;
    CUdeviceptr _textureObjectBuffer;
    CUtexObject _textureObject;
    CUdeviceptr _outputBuffer;
    CUevent _eventStart;
    CUevent _eventStop;
    size_t _width;
    size_t _pitchInBytes;
    size_t _height;
    size_t _blockSizeX;
    size_t _blockSizeY;
public:
    variant2D_cuda_impl_pitched_intermediate_memory(
        CUfunction kernel,
        size_t width,
        size_t height,
        std::span<const float> in,
        size_t inPitchInBytes,
        size_t blockSizeX,
        size_t blockSizeY
    )
    : _kernel(kernel)
    , _width(width)
    , _height(height)
    , _blockSizeX(blockSizeX)
    , _blockSizeY(blockSizeY)
    {
        if (_height * inPitchInBytes > in.size_bytes())
            throw std::exception("height * inPitchInBytes > in.size_bytes()!");

        check_error(cuMemAllocPitch(&_outputBuffer, &_pitchInBytes, 4 * sizeof(uint16_t) * width, height, 4 * sizeof(uint16_t)));

        size_t pitchInBytesTexture;
        check_error(cuMemAllocPitch(&_textureObjectBuffer, &pitchInBytesTexture, 4 * sizeof(float) * width, height, 4 * sizeof(float)));
        CUDA_RESOURCE_DESC texResDesc{};
        texResDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
        texResDesc.res.pitch2D.devPtr = _textureObjectBuffer;
        texResDesc.res.pitch2D.width = _width;
        texResDesc.res.pitch2D.height = _height;
        texResDesc.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
        texResDesc.res.pitch2D.numChannels = 4;
        texResDesc.res.pitch2D.pitchInBytes = pitchInBytesTexture;
        CUDA_TEXTURE_DESC texDesc{};
        texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        check_error(cuTexObjectCreate(&_textureObject, &texResDesc, &texDesc, nullptr));

        CUDA_MEMCPY2D memcpy2D{};
        memcpy2D.srcMemoryType = CU_MEMORYTYPE_HOST;
        memcpy2D.srcHost = in.data();
        memcpy2D.srcPitch = inPitchInBytes;
        memcpy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy2D.dstDevice = _textureObjectBuffer;
        memcpy2D.dstPitch = pitchInBytesTexture;
        memcpy2D.WidthInBytes = 4 * sizeof(float) * _width;
        memcpy2D.Height = _height;
        check_error(cuMemcpy2D(&memcpy2D));

        check_error(cuEventCreate(&_eventStart, 0));
        check_error(cuEventCreate(&_eventStop, 0));
    }
public:
    ~variant2D_cuda_impl_pitched_intermediate_memory() override
    {
        cuEventDestroy(_eventStart);
        cuEventDestroy(_eventStop);

        cuMemFree(_outputBuffer);
        cuTexObjectDestroy(_textureObject);
        cuMemFree(_textureObjectBuffer);
    }
public:
    variant_result execute(std::span<uint16_t> out, size_t outPitchInBytes) override
    {
        if (_height * outPitchInBytes > out.size_bytes())
            throw std::exception("height * outPitchInBytes > out.size_bytes()!");

        size_t gridSizeX = (_width + _blockSizeX * xBatchSize - 1) / (_blockSizeX * xBatchSize);
        size_t gridSizeY = (_height + _blockSizeY - 1) / _blockSizeY;
        void *args[] = {
            &_width,
            &_height,
            &_textureObject,
            &_outputBuffer,
            &_pitchInBytes
        };
        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuLaunchKernel(
            _kernel,
            gridSizeX, gridSizeY, 1,
            _blockSizeX, _blockSizeY, 1,
            0,
            nullptr,
            args,
            nullptr
        ));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float executeMilliseconds;
        check_error(cuEventElapsedTime(&executeMilliseconds, _eventStart, _eventStop));

        CUDA_MEMCPY2D memcpy2D{};
        memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy2D.srcDevice = _outputBuffer;
        memcpy2D.srcPitch = _pitchInBytes;
        memcpy2D.dstMemoryType = CU_MEMORYTYPE_HOST;
        memcpy2D.dstHost = out.data();
        memcpy2D.dstPitch = outPitchInBytes;
        memcpy2D.WidthInBytes = 4 * sizeof(uint16_t) * _width;
        memcpy2D.Height = _height;
        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpy2D(&memcpy2D));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float downloadMilliseconds;
        check_error(cuEventElapsedTime(&downloadMilliseconds, _eventStart, _eventStop));

        return {
            ._execute = std::chrono::nanoseconds{static_cast<long long>(executeMilliseconds * 1e6f)},
            ._download = std::chrono::nanoseconds{static_cast<long long>(downloadMilliseconds * 1e6f)},
        };
    }
};

class memory_layout_overhead_cuda_impl
{
    friend class memory_layout_overhead_cuda;
private:
    CUdevice _device;
    CUcontext _context;
    std::vector<CUmodule> _modules;
    CUfunction _non_pitched;
    CUfunction _batched_2_non_pitched;
    CUfunction _batched_4_non_pitched;
    CUfunction _batched_8_non_pitched;
    CUfunction _batched_16_non_pitched;
    CUfunction _batched_32_non_pitched;
    CUfunction _pitched;
    CUfunction _batched_2_pitched;
    CUfunction _batched_4_pitched;
    CUfunction _batched_8_pitched;
    CUfunction _batched_16_pitched;
    CUfunction _batched_32_pitched;
public:
    memory_layout_overhead_cuda_impl(const std::span<const std::filesystem::path> &modules)
    {
        check_error(cuInit(0));

        int deviceCount;
        check_error(cuDeviceGetCount(&deviceCount));
        if (deviceCount == 0)
            throw std::exception("No CUDA device found!");

        check_error(cuDeviceGet(&_device, 0));
        constexpr size_t maxNameLength = 256;
        char name[maxNameLength];
        cuDeviceGetName(name, maxNameLength - 1, _device);
        name[maxNameLength - 1] = '\0';
        std::cout << "Using device: " << name << std::endl;

        check_error(cuCtxCreate(&_context, 0, _device));

        try
        {
            std::transform(std::begin(modules), std::end(modules), std::back_inserter(_modules),
                           [](const std::filesystem::path path)
                           {
                               CUmodule module;
                               auto pathStr = path.string();
                               check_error(cuModuleLoad(&module, pathStr.c_str()));
                               return module;
                           }
            );

            _non_pitched = find_in_modules(_modules, "readback_texture");
            _batched_2_non_pitched = find_in_modules(_modules, "readback_texture_batched_2");
            _batched_4_non_pitched = find_in_modules(_modules, "readback_texture_batched_4");
            _batched_8_non_pitched = find_in_modules(_modules, "readback_texture_batched_8");
            _batched_16_non_pitched = find_in_modules(_modules, "readback_texture_batched_16");
            _batched_32_non_pitched = find_in_modules(_modules, "readback_texture_batched_32");
            _pitched = find_in_modules(_modules, "readback_texture_pitched_output");
            _batched_2_pitched = find_in_modules(_modules, "readback_texture_batched_2_pitched_output");
            _batched_4_pitched = find_in_modules(_modules, "readback_texture_batched_4_pitched_output");
            _batched_8_pitched = find_in_modules(_modules, "readback_texture_batched_8_pitched_output");
            _batched_16_pitched = find_in_modules(_modules, "readback_texture_batched_16_pitched_output");
            _batched_32_pitched = find_in_modules(_modules, "readback_texture_batched_32_pitched_output");
        }
        catch (const std::exception &exception)
        {
            cuCtxDestroy(_context);
            std::rethrow_exception(std::current_exception());
        }
    }
public:
    [[maybe_unused]] [[nodiscard]] std::vector<variant_id> variants2D() const
    {
        return {
            {0},
            {1},
            {2},
            {3},

            {4},
            {5},
            {6},
            {7},

            {8},
            {9},
            {10},
            {11},

            {12},
            {13},
            {14},
            {15},

            {16},
            {17},
            {18},
            {19},

            {20},
            {21},
            {22},
            {23},

            {32},
            {33},
            {34},
            {35},

            {36},
            {37},
            {38},
            {39},

            {40},
            {41},
            {42},
            {43},

            {44},
            {45},
            {46},
            {47},

            {48},
            {49},
            {50},
            {51},

            {52},
            {53},
            {54},
            {55},
        };
    }

    [[maybe_unused]] [[nodiscard]] std::string_view variant_name(const variant_id &id) const
    {
        switch (id.id)
        {
            case 0:
                return "readback_texture_non_pitched_output [1, 1]";
            case 1:
                return "readback_texture_non_pitched_output [4, 4]";
            case 2:
                return "readback_texture_non_pitched_output [8, 8]";
            case 3:
                return "readback_texture_non_pitched_output [16, 16]";

            case 4:
                return "readback_texture_batched_2_non_pitched_output [1, 1]";
            case 5:
                return "readback_texture_batched_2_non_pitched_output [4, 4]";
            case 6:
                return "readback_texture_batched_2_non_pitched_output [8, 8]";
            case 7:
                return "readback_texture_batched_2_non_pitched_output [16, 16]";

            case 8:
                return "readback_texture_batched_4_non_pitched_output [1, 1]";
            case 9:
                return "readback_texture_batched_4_non_pitched_output [4, 4]";
            case 10:
                return "readback_texture_batched_4_non_pitched_output [8, 8]";
            case 11:
                return "readback_texture_batched_4_non_pitched_output [16, 16]";

            case 12:
                return "readback_texture_batched_8_non_pitched_output [1, 1]";
            case 13:
                return "readback_texture_batched_8_non_pitched_output [4, 4]";
            case 14:
                return "readback_texture_batched_8_non_pitched_output [8, 8]";
            case 15:
                return "readback_texture_batched_8_non_pitched_output [16, 16]";

            case 16:
                return "readback_texture_batched_16_non_pitched_output [1, 1]";
            case 17:
                return "readback_texture_batched_16_non_pitched_output [4, 4]";
            case 18:
                return "readback_texture_batched_16_non_pitched_output [8, 8]";
            case 19:
                return "readback_texture_batched_16_non_pitched_output [16, 16]";

            case 20:
                return "readback_texture_batched_32_non_pitched_output [1, 1]";
            case 21:
                return "readback_texture_batched_32_non_pitched_output [4, 4]";
            case 22:
                return "readback_texture_batched_32_non_pitched_output [8, 8]";
            case 23:
                return "readback_texture_batched_32_non_pitched_output [16, 16]";

            case 32:
                return "readback_texture_pitched_output [1, 1]";
            case 33:
                return "readback_texture_pitched_output [4, 4]";
            case 34:
                return "readback_texture_pitched_output [8, 8]";
            case 35:
                return "readback_texture_pitched_output [16, 16]";

            case 36:
                return "readback_texture_batched_2_pitched_output [1, 1]";
            case 37:
                return "readback_texture_batched_2_pitched_output [4, 4]";
            case 38:
                return "readback_texture_batched_2_pitched_output [8, 8]";
            case 39:
                return "readback_texture_batched_2_pitched_output [16, 16]";

            case 40:
                return "readback_texture_batched_4_pitched_output [1, 1]";
            case 41:
                return "readback_texture_batched_4_pitched_output [4, 4]";
            case 42:
                return "readback_texture_batched_4_pitched_output [8, 8]";
            case 43:
                return "readback_texture_batched_4_pitched_output [16, 16]";

            case 44:
                return "readback_texture_batched_8_pitched_output [1, 1]";
            case 45:
                return "readback_texture_batched_8_pitched_output [4, 4]";
            case 46:
                return "readback_texture_batched_8_pitched_output [8, 8]";
            case 47:
                return "readback_texture_batched_8_pitched_output [16, 16]";

            case 48:
                return "readback_texture_batched_16_pitched_output [1, 1]";
            case 49:
                return "readback_texture_batched_16_pitched_output [4, 4]";
            case 50:
                return "readback_texture_batched_16_pitched_output [8, 8]";
            case 51:
                return "readback_texture_batched_16_pitched_output [16, 16]";

            case 52:
                return "readback_texture_batched_16_pitched_output [1, 1]";
            case 53:
                return "readback_texture_batched_16_pitched_output [4, 4]";
            case 54:
                return "readback_texture_batched_16_pitched_output [8, 8]";
            case 55:
                return "readback_texture_batched_16_pitched_output [16, 16]";

            default:
                throw std::exception("Invalid variant!");
        }
    }
public:
    std::shared_ptr<variant2D_impl> create_variant2D_impl(const variant_id &id, size_t width, size_t height, std::span<const float> in, size_t inPitchInBytes)
    {
        switch (id.id)
        {
            case 0:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<1>>(_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 1, 1
                );
            case 1:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<1>>(_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 4, 4
                );
            case 2:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<1>>(_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 8, 8
                );
            case 3:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<1>>(_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 16, 16
                );
            case 4:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<2>>(_batched_2_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 1, 1
                );
            case 5:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<2>>(_batched_2_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 4, 4
                );
            case 6:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<2>>(_batched_2_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 8, 8
                );
            case 7:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<2>>(_batched_2_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 16, 16
                );
            case 8:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<4>>(_batched_4_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 1, 1
                );
            case 9:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<4>>(_batched_4_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 4, 4
                );
            case 10:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<4>>(_batched_4_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 8, 8
                );
            case 11:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<4>>(_batched_4_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 16, 16
                );
            case 12:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<8>>(_batched_8_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 1, 1
                );
            case 13:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<8>>(_batched_8_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 4, 4
                );
            case 14:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<8>>(_batched_8_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 8, 8
                );
            case 15:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<8>>(_batched_8_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 16, 16
                );
            case 16:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<16>>(_batched_16_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 1, 1
                );
            case 17:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<16>>(_batched_16_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 4, 4
                );
            case 18:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<16>>(_batched_16_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 8, 8
                );
            case 19:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<16>>(_batched_16_non_pitched, width, height, in,
                                                                                                inPitchInBytes, 16, 16
                );
            case 20:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<32>>(_batched_32_non_pitched, width, height, in,
                                                                                                 inPitchInBytes, 1, 1
                );
            case 21:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<32>>(_batched_32_non_pitched, width, height, in,
                                                                                                 inPitchInBytes, 4, 4
                );
            case 22:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<32>>(_batched_32_non_pitched, width, height, in,
                                                                                                 inPitchInBytes, 8, 8
                );
            case 23:
                return std::make_shared<variant2D_cuda_impl_non_pitched_intermediate_memory<32>>(_batched_32_non_pitched, width, height, in,
                                                                                                 inPitchInBytes, 16, 16
                );

            case 32:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<1>>(_pitched, width, height, in,
                                                                                            inPitchInBytes, 1, 1
                );
            case 33:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<1>>(_pitched, width, height, in,
                                                                                            inPitchInBytes, 4, 4
                );
            case 34:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<1>>(_pitched, width, height, in,
                                                                                            inPitchInBytes, 8, 8
                );
            case 35:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<1>>(_pitched, width, height, in,
                                                                                            inPitchInBytes, 16, 16
                );

            case 36:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<2>>(_batched_2_pitched, width, height, in,
                                                                                            inPitchInBytes, 1, 1
                );
            case 37:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<2>>(_batched_2_pitched, width, height, in,
                                                                                            inPitchInBytes, 4, 4
                );
            case 38:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<2>>(_batched_2_pitched, width, height, in,
                                                                                            inPitchInBytes, 8, 8
                );
            case 39:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<2>>(_batched_2_pitched, width, height, in,
                                                                                            inPitchInBytes, 16, 16
                );

            case 40:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<4>>(_batched_4_pitched, width, height, in,
                                                                                            inPitchInBytes, 1, 1
                );
            case 41:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<4>>(_batched_4_pitched, width, height, in,
                                                                                            inPitchInBytes, 4, 4
                );
            case 42:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<4>>(_batched_4_pitched, width, height, in,
                                                                                            inPitchInBytes, 8, 8
                );
            case 43:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<4>>(_batched_4_pitched, width, height, in,
                                                                                            inPitchInBytes, 16, 16
                );

            case 44:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<8>>(_batched_8_pitched, width, height, in,
                                                                                            inPitchInBytes, 1, 1
                );
            case 45:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<8>>(_batched_8_pitched, width, height, in,
                                                                                            inPitchInBytes, 4, 4
                );
            case 46:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<8>>(_batched_8_pitched, width, height, in,
                                                                                            inPitchInBytes, 8, 8
                );
            case 47:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<8>>(_batched_8_pitched, width, height, in,
                                                                                            inPitchInBytes, 16, 16
                );

            case 48:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<16>>(_batched_16_pitched, width, height, in,
                                                                                            inPitchInBytes, 1, 1
                );
            case 49:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<16>>(_batched_16_pitched, width, height, in,
                                                                                            inPitchInBytes, 4, 4
                );
            case 50:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<16>>(_batched_16_pitched, width, height, in,
                                                                                            inPitchInBytes, 8, 8
                );
            case 51:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<16>>(_batched_16_pitched, width, height, in,
                                                                                            inPitchInBytes, 16, 16
                );

            case 52:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<32>>(_batched_32_pitched, width, height, in,
                                                                                             inPitchInBytes, 1, 1
                );
            case 53:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<32>>(_batched_32_pitched, width, height, in,
                                                                                             inPitchInBytes, 4, 4
                );
            case 54:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<32>>(_batched_32_pitched, width, height, in,
                                                                                             inPitchInBytes, 8, 8
                );
            case 55:
                return std::make_shared<variant2D_cuda_impl_pitched_intermediate_memory<32>>(_batched_32_pitched, width, height, in,
                                                                                             inPitchInBytes, 16, 16
                );

            default:
                throw std::exception("Invalid variant!");
        }
    }
public:
    ~memory_layout_overhead_cuda_impl()
    {
        cuCtxDestroy(_context);
    }
};

memory_layout_overhead_cuda::memory_layout_overhead_cuda(const std::span<const std::filesystem::path> &modules)
    : _impl(std::make_shared<memory_layout_overhead_cuda_impl>(modules))
{}

std::vector<variant_id> memory_layout_overhead_cuda::variants2D() const
{
    return _impl->variants2D();
}

std::string_view memory_layout_overhead_cuda::variant_name(const variant_id &id) const
{
    return _impl->variant_name(id);
}

std::shared_ptr<variant2D_impl>
memory_layout_overhead_cuda::create_variant2D_impl(const variant_id &id, size_t width, size_t height, std::span<const float> in, size_t inPitchInBytes)
{
    return _impl->create_variant2D_impl(id, width, height, in, inPitchInBytes);
}

}