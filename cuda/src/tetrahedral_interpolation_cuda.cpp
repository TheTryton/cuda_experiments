#include <tetrahedral_interpolation_cuda.hpp>

#include <cuda_helpers.hpp>

namespace tetrahedral_interpolation
{

class lut3D_cuda_impl
    : public lut3D_impl
{
    friend class variant1D_cuda_impl_raw_in_raw_out;
    friend class variant2D_cuda_impl_raw_in_raw_out;
    friend class variant2D_cuda_impl_texture_in_raw_out;
    friend class variant2D_cuda_impl_texture_in_surface_out;
private:
    CUarray _array;
    CUtexObject _lut3D;
    size_t _lutSize;
public:
    ~lut3D_cuda_impl() override
    {
        cuArrayDestroy(_array);
        cuTexObjectDestroy(_lut3D);
    }
public:
    lut3D_cuda_impl(const void *data, size_t lutSize)
        : _lutSize(lutSize)
    {
        CUDA_ARRAY3D_DESCRIPTOR desc{
            .Width = lutSize,
            .Height = lutSize,
            .Depth = lutSize,
            .Format = CUarray_format::CU_AD_FORMAT_FLOAT,
            .NumChannels = 4,
            .Flags = 0
        };

        check_error(cuArray3DCreate(&_array, &desc));

        try
        {
            CUDA_MEMCPY3D memcpy3D{};
            memcpy3D.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
            memcpy3D.srcHost = data;
            memcpy3D.srcPitch = lutSize * 4 * sizeof(float);
            memcpy3D.srcHeight = lutSize;
            memcpy3D.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
            memcpy3D.dstArray = _array;
            memcpy3D.WidthInBytes = lutSize * 4 * sizeof(float);
            memcpy3D.Height = lutSize;
            memcpy3D.Depth = lutSize;

            check_error(cuMemcpy3D(&memcpy3D));

            CUDA_RESOURCE_DESC texRes{};
            texRes.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
            texRes.res.array.hArray = _array;

            CUDA_TEXTURE_DESC texDesc{};
            texDesc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
            texDesc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
            texDesc.addressMode[2] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
            texDesc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_POINT;

            check_error(cuTexObjectCreate(&_lut3D, &texRes, &texDesc, nullptr));
        }
        catch (...)
        {
            cuArrayDestroy(_array);
            std::rethrow_exception(std::current_exception());
        }
    }
};

class variant1D_cuda_impl_raw_in_raw_out
    : public variant1D_impl
{
private:
    CUfunction _kernel;
    CUdeviceptr _in;
    CUdeviceptr _out;
    CUevent _eventStart;
    CUevent _eventStop;
    size_t _size;
    size_t _blockSize;
public:
    variant1D_cuda_impl_raw_in_raw_out(CUfunction kernel, size_t size, size_t blockSize)
        : _kernel(kernel)
          , _size(size)
          , _blockSize(blockSize)
    {
        check_error(cuMemAlloc(&_in, _size * 4 * sizeof(uint8_t)));
        check_error(cuMemAlloc(&_out, _size * 4 * sizeof(uint8_t)));

        check_error(cuEventCreate(&_eventStart, 0));
        check_error(cuEventCreate(&_eventStop, 0));
    }
public:
    ~variant1D_cuda_impl_raw_in_raw_out() override
    {
        cuEventDestroy(_eventStart);
        cuEventDestroy(_eventStop);

        cuMemFree(_in);
        cuMemFree(_out);
    }
public:
    variant_result execute(const void *in, void *out, const lut3D_impl &lut3D_) override
    {
        auto lut3D = dynamic_cast<const lut3D_cuda_impl *>(&lut3D_);
        if (lut3D == nullptr)
            throw std::exception("Invalid LUT!");

        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpyHtoD(_in, in, _size * 4 * sizeof(uint8_t)));
        check_error(cuEventRecord(_eventStop, nullptr));
        float uploadMilliseconds;
        check_error(cuEventSynchronize(_eventStop));
        check_error(cuEventElapsedTime(&uploadMilliseconds, _eventStart, _eventStop));

        check_error(cuEventRecord(_eventStart, nullptr));
        size_t gridSize = (_size + _blockSize - 1) / _blockSize;
        void *args[] = {
            &_in,
            &_out,
            &_size,
            const_cast<void *>(static_cast<const void *>(&lut3D->_lut3D)),
            const_cast<void *>(static_cast<const void *>(&lut3D->_lutSize)),
        };
        check_error(cuLaunchKernel(
            _kernel,
            gridSize, 1, 1,
            _blockSize, 1, 1,
            0,
            nullptr,
            args,
            nullptr
        ));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float executeMilliseconds;
        check_error(cuEventElapsedTime(&executeMilliseconds, _eventStart, _eventStop));

        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpyDtoH(out, _out, _size * 4 * sizeof(uint8_t)));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float downloadMilliseconds;
        check_error(cuEventElapsedTime(&downloadMilliseconds, _eventStart, _eventStop));

        return {
            ._upload = std::chrono::nanoseconds{static_cast<long long>(uploadMilliseconds * 1e6f)},
            ._execute = std::chrono::nanoseconds{static_cast<long long>(executeMilliseconds * 1e6f)},
            ._download = std::chrono::nanoseconds{static_cast<long long>(downloadMilliseconds * 1e6f)},
        };
    }
};

class variant2D_cuda_impl_raw_in_raw_out
    : public variant2D_impl
{
private:
    CUfunction _kernel;
    CUdeviceptr _in;
    CUdeviceptr _out;
    CUevent _eventStart;
    CUevent _eventStop;
    size_t _width;
    size_t _height;
    size_t _blockSizeX;
    size_t _blockSizeY;
public:
    variant2D_cuda_impl_raw_in_raw_out(
        CUfunction kernel,
        size_t width,
        size_t height,
        size_t blockSizeX,
        size_t blockSizeY
    )
        : _kernel(kernel)
          , _width(width)
          , _height(height)
          , _blockSizeX(blockSizeX)
          , _blockSizeY(blockSizeY)
    {
        check_error(cuMemAlloc(&_in, _width * _height * 4 * sizeof(uint8_t)));
        check_error(cuMemAlloc(&_out, _width * _height * 4 * sizeof(uint8_t)));

        check_error(cuEventCreate(&_eventStart, 0));
        check_error(cuEventCreate(&_eventStop, 0));
    }
public:
    ~variant2D_cuda_impl_raw_in_raw_out() override
    {
        cuEventDestroy(_eventStart);
        cuEventDestroy(_eventStop);

        cuMemFree(_in);
        cuMemFree(_out);
    }
public:
    variant_result execute(const void *in, void *out, const lut3D_impl &lut3D_) override
    {
        auto lut3D = dynamic_cast<const lut3D_cuda_impl *>(&lut3D_);
        if (lut3D == nullptr)
            throw std::exception("Invalid LUT!");

        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpyHtoD(_in, in, _width * _height * 4 * sizeof(uint8_t)));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float uploadMilliseconds;
        check_error(cuEventElapsedTime(&uploadMilliseconds, _eventStart, _eventStop));

        check_error(cuEventRecord(_eventStart, nullptr));
        size_t gridSizeX = (_width + _blockSizeX - 1) / _blockSizeX;
        size_t gridSizeY = (_height + _blockSizeY - 1) / _blockSizeY;
        void *args[] = {
            &_in,
            &_out,
            &_width,
            &_height,
            const_cast<void *>(static_cast<const void *>(&lut3D->_lut3D)),
            const_cast<void *>(static_cast<const void *>(&lut3D->_lutSize)),
        };
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

        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpyDtoH(out, _out, _width * _height * 4 * sizeof(uint8_t)));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float downloadMilliseconds;
        check_error(cuEventElapsedTime(&downloadMilliseconds, _eventStart, _eventStop));

        return {
            ._upload = std::chrono::nanoseconds{static_cast<long long>(uploadMilliseconds * 1e6f)},
            ._execute = std::chrono::nanoseconds{static_cast<long long>(executeMilliseconds * 1e6f)},
            ._download = std::chrono::nanoseconds{static_cast<long long>(downloadMilliseconds * 1e6f)},
        };
    }
};

class variant2D_cuda_impl_texture_in_raw_out
    : public variant2D_impl
{
private:
    CUfunction _kernel;
    CUarray _in;
    CUtexObject _inTexture;
    CUdeviceptr _out;
    CUevent _eventStart;
    CUevent _eventStop;
    size_t _width;
    size_t _height;
    size_t _blockSizeX;
    size_t _blockSizeY;
public:
    variant2D_cuda_impl_texture_in_raw_out(
        CUfunction kernel,
        size_t width,
        size_t height,
        size_t blockSizeX,
        size_t blockSizeY
    )
        : _kernel(kernel)
          , _width(width)
          , _height(height)
          , _blockSizeX(blockSizeX)
          , _blockSizeY(blockSizeY)
    {
        CUDA_ARRAY_DESCRIPTOR arrayDesc{};
        arrayDesc.Width = _width;
        arrayDesc.Height = _height;
        arrayDesc.Format = CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8;
        arrayDesc.NumChannels = 4;

        check_error(cuArrayCreate(&_in, &arrayDesc));

        check_error(cuMemAlloc(&_out, _width * _height * 4 * sizeof(uint8_t)));

        CUDA_RESOURCE_DESC texRes{};
        texRes.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        texRes.res.array.hArray = _in;

        CUDA_TEXTURE_DESC texDesc{};
        texDesc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[2] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_POINT;

        check_error(cuTexObjectCreate(&_inTexture, &texRes, &texDesc, nullptr));

        check_error(cuEventCreate(&_eventStart, 0));
        check_error(cuEventCreate(&_eventStop, 0));
    }
public:
    ~variant2D_cuda_impl_texture_in_raw_out() override
    {
        cuEventDestroy(_eventStart);
        cuEventDestroy(_eventStop);

        cuTexObjectDestroy(_inTexture);

        cuArrayDestroy(_in);
        cuMemFree(_out);
    }
public:
    variant_result execute(const void *in, void *out, const lut3D_impl &lut3D_) override
    {
        auto lut3D = dynamic_cast<const lut3D_cuda_impl *>(&lut3D_);
        if (lut3D == nullptr)
            throw std::exception("Invalid LUT!");

        check_error(cuEventRecord(_eventStart, nullptr));
        CUDA_MEMCPY2D memcpy2D{};
        memcpy2D.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
        memcpy2D.srcHost = in;
        memcpy2D.srcPitch = _width * 4 * sizeof(uint8_t);
        memcpy2D.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
        memcpy2D.dstArray = _in;
        memcpy2D.WidthInBytes = _width * 4 * sizeof(uint8_t);
        memcpy2D.Height = _height;
        check_error(cuMemcpy2D(&memcpy2D));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float uploadMilliseconds;
        check_error(cuEventElapsedTime(&uploadMilliseconds, _eventStart, _eventStop));

        check_error(cuEventRecord(_eventStart, nullptr));
        size_t gridSizeX = (_width + _blockSizeX - 1) / _blockSizeX;
        size_t gridSizeY = (_height + _blockSizeY - 1) / _blockSizeY;
        void *args[] = {
            &_inTexture,
            &_out,
            &_width,
            &_height,
            const_cast<void *>(static_cast<const void *>(&lut3D->_lut3D)),
            const_cast<void *>(static_cast<const void *>(&lut3D->_lutSize)),
        };
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

        check_error(cuEventRecord(_eventStart, nullptr));
        check_error(cuMemcpyDtoH(out, _out, _width * _height * 4 * sizeof(uint8_t)));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float downloadMilliseconds;
        check_error(cuEventElapsedTime(&downloadMilliseconds, _eventStart, _eventStop));

        return {
            ._upload = std::chrono::nanoseconds{static_cast<long long>(uploadMilliseconds * 1e6f)},
            ._execute = std::chrono::nanoseconds{static_cast<long long>(executeMilliseconds * 1e6f)},
            ._download = std::chrono::nanoseconds{static_cast<long long>(downloadMilliseconds * 1e6f)},
        };
    }
};

class variant2D_cuda_impl_texture_in_surface_out
    : public variant2D_impl
{
private:
    CUfunction _kernel;
    CUarray _in;
    CUtexObject _inTexture;
    CUarray _out;
    CUsurfObject _outSurface;
    CUevent _eventStart;
    CUevent _eventStop;
    size_t _width;
    size_t _height;
    size_t _blockSizeX;
    size_t _blockSizeY;
public:
    variant2D_cuda_impl_texture_in_surface_out(
        CUfunction kernel,
        size_t width,
        size_t height,
        size_t blockSizeX,
        size_t blockSizeY
    )
        : _kernel(kernel)
          , _width(width)
          , _height(height)
          , _blockSizeX(blockSizeX)
          , _blockSizeY(blockSizeY)
    {
        CUDA_ARRAY_DESCRIPTOR arrayDescIn{};
        arrayDescIn.Width = _width;
        arrayDescIn.Height = _height;
        arrayDescIn.Format = CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8;
        arrayDescIn.NumChannels = 4;

        check_error(cuArrayCreate(&_in, &arrayDescIn));

        CUDA_ARRAY3D_DESCRIPTOR arrayDescOut{};
        arrayDescOut.Width = _width;
        arrayDescOut.Height = _height;
        arrayDescOut.Format = CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8;
        arrayDescOut.NumChannels = 4;
        arrayDescOut.Flags = CUDA_ARRAY3D_SURFACE_LDST;
        check_error(cuArray3DCreate(&_out, &arrayDescOut));

        CUDA_RESOURCE_DESC texRes{};
        texRes.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        texRes.res.array.hArray = _in;

        CUDA_TEXTURE_DESC texDesc{};
        texDesc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[2] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_POINT;

        check_error(cuTexObjectCreate(&_inTexture, &texRes, &texDesc, nullptr));

        CUDA_RESOURCE_DESC surfRes{};
        surfRes.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        surfRes.res.array.hArray = _out;

        check_error(cuSurfObjectCreate(&_outSurface, &surfRes));

        check_error(cuEventCreate(&_eventStart, 0));
        check_error(cuEventCreate(&_eventStop, 0));
    }
public:
    ~variant2D_cuda_impl_texture_in_surface_out() override
    {
        cuEventDestroy(_eventStart);
        cuEventDestroy(_eventStop);

        cuSurfObjectDestroy(_outSurface);
        cuTexObjectDestroy(_inTexture);

        cuArrayDestroy(_in);
        cuArrayDestroy(_out);
    }
public:
    variant_result execute(const void *in, void *out, const lut3D_impl &lut3D_) override
    {
        auto lut3D = dynamic_cast<const lut3D_cuda_impl *>(&lut3D_);
        if (lut3D == nullptr)
            throw std::exception("Invalid LUT!");

        check_error(cuEventRecord(_eventStart, nullptr));
        CUDA_MEMCPY2D memcpy2Din{};
        memcpy2Din.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
        memcpy2Din.srcHost = in;
        memcpy2Din.srcPitch = _width * 4 * sizeof(uint8_t);
        memcpy2Din.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
        memcpy2Din.dstArray = _in;
        memcpy2Din.WidthInBytes = _width * 4 * sizeof(uint8_t);
        memcpy2Din.Height = _height;
        check_error(cuMemcpy2D(&memcpy2Din));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float uploadMilliseconds;
        check_error(cuEventElapsedTime(&uploadMilliseconds, _eventStart, _eventStop));

        check_error(cuEventRecord(_eventStart, nullptr));
        size_t gridSizeX = (_width + _blockSizeX - 1) / _blockSizeX;
        size_t gridSizeY = (_height + _blockSizeY - 1) / _blockSizeY;
        void *args[] = {
            &_inTexture,
            &_outSurface,
            &_width,
            &_height,
            const_cast<void *>(static_cast<const void *>(&lut3D->_lut3D)),
            const_cast<void *>(static_cast<const void *>(&lut3D->_lutSize)),
        };
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

        check_error(cuEventRecord(_eventStart, nullptr));
        CUDA_MEMCPY2D memcpy2Dout{};
        memcpy2Dout.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
        memcpy2Dout.srcArray = _out;
        memcpy2Dout.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
        memcpy2Dout.dstHost = out;
        memcpy2Dout.dstPitch = _width * 4 * sizeof(uint8_t);
        memcpy2Dout.WidthInBytes = _width * 4 * sizeof(uint8_t);
        memcpy2Dout.Height = _height;
        check_error(cuMemcpy2D(&memcpy2Dout));
        check_error(cuEventRecord(_eventStop, nullptr));
        check_error(cuEventSynchronize(_eventStop));
        float downloadMilliseconds;
        check_error(cuEventElapsedTime(&downloadMilliseconds, _eventStart, _eventStop));

        return {
            ._upload = std::chrono::nanoseconds{static_cast<long long>(uploadMilliseconds * 1e6f)},
            ._execute = std::chrono::nanoseconds{static_cast<long long>(executeMilliseconds * 1e6f)},
            ._download = std::chrono::nanoseconds{static_cast<long long>(downloadMilliseconds * 1e6f)},
        };
    }
};

class tetrahedral_interpolation_cuda_impl
{
    friend class tetrahedral_interpolation_cuda;
private:
    CUdevice _device;
    CUcontext _context;
    std::vector<CUmodule> _modules;
    CUfunction _1D_raw_in_raw_out;
    CUfunction _2D_raw_in_raw_out;
    CUfunction _2D_texture_in_raw_out;
    CUfunction _2D_texture_in_surface_out;
public:
    tetrahedral_interpolation_cuda_impl(const std::span<const std::filesystem::path> &modules)
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

            _1D_raw_in_raw_out = find_in_modules(_modules, "apply_lut3D_R8G8B8A8_1D_raw_in_raw_out");
            _2D_raw_in_raw_out = find_in_modules(_modules, "apply_lut3D_R8G8B8A8_2D_raw_in_raw_out");
            _2D_texture_in_raw_out = find_in_modules(_modules, "apply_lut3D_R8G8B8A8_2D_texture_in_raw_out");
            _2D_texture_in_surface_out = find_in_modules(_modules, "apply_lut3D_R8G8B8A8_2D_texture_in_surface_out");
        }
        catch (const std::exception &exception)
        {
            cuCtxDestroy(_context);
            std::rethrow_exception(std::current_exception());
        }
    }
public:
    ~tetrahedral_interpolation_cuda_impl()
    {
        cuCtxDestroy(_context);
    }
public:
    [[maybe_unused]] [[nodiscard]] std::vector<variant_id> variants1D() const
    {
        return {
            {0},
            {1},
            {2},
            {3},
        };
    }
    [[maybe_unused]] [[nodiscard]] std::vector<variant_id> variants2D() const
    {
        return {
            {32},
            {33},
            {34},
            {35},
            {64},
            {65},
            {66},
            {67},
            {96},
            {97},
            {98},
            {99},
        };
    }

    [[maybe_unused]] [[nodiscard]] std::string_view variant_name(const variant_id &id) const
    {
        switch (id.id)
        {
            case 0:
                return "raw_in_raw_out [1]";
            case 1:
                return "raw_in_raw_out [4]";
            case 2:
                return "raw_in_raw_out [16]";
            case 3:
                return "raw_in_raw_out [128]";

            case 32:
                return "raw_in_raw_out [1, 1]";
            case 33:
                return "raw_in_raw_out [4, 4]";
            case 34:
                return "raw_in_raw_out [8, 8]";
            case 35:
                return "raw_in_raw_out [16, 16]";

            case 64:
                return "texture_in_raw_out [1, 1]";
            case 65:
                return "texture_in_raw_out [4, 4]";
            case 66:
                return "texture_in_raw_out [8, 8]";
            case 67:
                return "texture_in_raw_out [16, 16]";

            case 96:
                return "texture_in_surface_out [1, 1]";
            case 97:
                return "texture_in_surface_out [4, 4]";
            case 98:
                return "texture_in_surface_out [8, 8]";
            case 99:
                return "texture_in_surface_out [16, 16]";

            default:
                throw std::exception("Invalid variant!");
        }
    }
protected:
    std::shared_ptr<lut3D_impl> create_lut3D_impl(const void *data, size_t lutSize)
    {
        return std::make_shared<lut3D_cuda_impl>(data, lutSize);
    }

    std::shared_ptr<variant1D_impl> create_variant1D_impl(const variant_id &id, size_t size)
    {
        switch (id.id)
        {
            case 0:
                return std::make_shared<variant1D_cuda_impl_raw_in_raw_out>(_1D_raw_in_raw_out, size, 1);
            case 1:
                return std::make_shared<variant1D_cuda_impl_raw_in_raw_out>(_1D_raw_in_raw_out, size, 4);
            case 2:
                return std::make_shared<variant1D_cuda_impl_raw_in_raw_out>(_1D_raw_in_raw_out, size, 16);
            case 3:
                return std::make_shared<variant1D_cuda_impl_raw_in_raw_out>(_1D_raw_in_raw_out, size, 128);

            default:
                throw std::exception("Invalid variant!");
        }
    }
    std::shared_ptr<variant2D_impl> create_variant2D_impl(const variant_id &id, size_t width, size_t height)
    {
        switch (id.id)
        {
            case 32:
                return std::make_shared<variant2D_cuda_impl_raw_in_raw_out>(_2D_raw_in_raw_out, width, height, 1, 1);
            case 33:
                return std::make_shared<variant2D_cuda_impl_raw_in_raw_out>(_2D_raw_in_raw_out, width, height, 4, 4);
            case 34:
                return std::make_shared<variant2D_cuda_impl_raw_in_raw_out>(_2D_raw_in_raw_out, width, height, 8, 8);
            case 35:
                return std::make_shared<variant2D_cuda_impl_raw_in_raw_out>(_2D_raw_in_raw_out, width, height, 16, 16);

            case 64:
                return std::make_shared<variant2D_cuda_impl_texture_in_raw_out>(_2D_texture_in_raw_out, width, height,
                                                                                1, 1
                );
            case 65:
                return std::make_shared<variant2D_cuda_impl_texture_in_raw_out>(_2D_texture_in_raw_out, width, height,
                                                                                4, 4
                );
            case 66:
                return std::make_shared<variant2D_cuda_impl_texture_in_raw_out>(_2D_texture_in_raw_out, width, height,
                                                                                8, 8
                );
            case 67:
                return std::make_shared<variant2D_cuda_impl_texture_in_raw_out>(_2D_texture_in_raw_out, width, height,
                                                                                16, 16
                );

            case 96:
                return std::make_shared<variant2D_cuda_impl_texture_in_surface_out>(_2D_texture_in_surface_out, width,
                                                                                    height, 1, 1
                );
            case 97:
                return std::make_shared<variant2D_cuda_impl_texture_in_surface_out>(_2D_texture_in_surface_out, width,
                                                                                    height, 4, 4
                );
            case 98:
                return std::make_shared<variant2D_cuda_impl_texture_in_surface_out>(_2D_texture_in_surface_out, width,
                                                                                    height, 8, 8
                );
            case 99:
                return std::make_shared<variant2D_cuda_impl_texture_in_surface_out>(_2D_texture_in_surface_out, width,
                                                                                    height, 16, 16
                );

            default:
                throw std::exception("Invalid variant!");
        }
    }
};

tetrahedral_interpolation_cuda::tetrahedral_interpolation_cuda(const std::span<const std::filesystem::path> &modules)
    : _impl(std::make_shared<tetrahedral_interpolation_cuda_impl>(modules))
{}

std::vector<variant_id> tetrahedral_interpolation_cuda::variants1D() const
{
    return _impl->variants1D();
}

std::vector<variant_id> tetrahedral_interpolation_cuda::variants2D() const
{
    return _impl->variants2D();
}

std::string_view tetrahedral_interpolation_cuda::variant_name(const variant_id &id) const
{
    return _impl->variant_name(id);
}

std::shared_ptr<lut3D_impl> tetrahedral_interpolation_cuda::create_lut3D_impl(const void *data, size_t lutSize)
{
    return _impl->create_lut3D_impl(data, lutSize);
}

std::shared_ptr<variant1D_impl> tetrahedral_interpolation_cuda::create_variant1D_impl(const variant_id &id, size_t size)
{
    return _impl->create_variant1D_impl(id, size);
}

std::shared_ptr<variant2D_impl>
tetrahedral_interpolation_cuda::create_variant2D_impl(const variant_id &id, size_t width, size_t height)
{
    return _impl->create_variant2D_impl(id, width, height);
}

}
