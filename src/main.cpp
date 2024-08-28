#include <tetrahedral_interpolation_cuda.hpp>
#include <memory_layout_overhead_cuda.hpp>
#include <cube_lut.hpp>

#include <iostream>
#include <fstream>
#include <random>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;
using namespace std::filesystem;
using namespace std::chrono;

#define BASE_DEBUG "C:\\Users\\michal\\Desktop\\tetrahedral_interpolation\\build\\msvc\\debug\\cuda\\kernels\\CMakeFiles\\tetrahedral_interpolation_cuda_kernels.dir\\src\\"
#define BASE_RELEASE "C:\\Users\\michal\\Desktop\\tetrahedral_interpolation\\build\\msvc\\release\\cuda\\kernels\\CMakeFiles\\tetrahedral_interpolation_cuda_kernels.dir\\src\\"

struct stbi_deleter
{
    void operator()(uint8_t* ptr)
    {
        if(ptr != nullptr)
            stbi_image_free(reinterpret_cast<stbi_uc*>(ptr));
    }
};

std::unique_ptr<uint8_t, stbi_deleter> load_image(const char* path, size_t& width, size_t& height)
{
    int w, h, channels;
    stbi_uc* image = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    width = w;
    height = h;

    return std::unique_ptr<uint8_t, stbi_deleter>(reinterpret_cast<uint8_t*>(image), {});
}

int save_image(const char* path, const uint8_t* data, size_t width, size_t height)
{
    return stbi_write_png(
        path,
        static_cast<int>(width),
        static_cast<int>(height),
        STBI_rgb_alpha,
        data,
        static_cast<int>(width * 4 * sizeof(stbi_uc))
    );
}

using milliseconds_double = duration<double, std::milli>;

tetrahedral_interpolation::variant_result calculate_avg_time_1D(
    tetrahedral_interpolation::tetrahedral_interpolation& tI,
    const variant_id& id,
    const void* in, void* out, size_t size,
    const tetrahedral_interpolation::lut3D& lut3D,
    size_t repeat = 100
)
{
    tetrahedral_interpolation::variant_result sum{};

    auto variant = tI.create_variant1D(id, size);

    for(size_t i = 0; i < repeat; i++)
    {
        const auto result = variant.execute(in, out, lut3D);
        sum._upload += result._upload;
        sum._execute += result._execute;
        sum._download += result._download;
    }

    return {
        ._upload = sum._upload / repeat,
        ._execute = sum._execute / repeat,
        ._download = sum._download / repeat,
    };
}

tetrahedral_interpolation::variant_result calculate_avg_time_2D(
    tetrahedral_interpolation::tetrahedral_interpolation& tI,
    const variant_id& id,
    const void* in, void* out, size_t width, size_t height,
    const tetrahedral_interpolation::lut3D& lut3D,
    size_t repeat = 100
)
{
    tetrahedral_interpolation::variant_result sum{};

    auto variant = tI.create_variant2D(id, width, height);

    for(size_t i = 0; i < repeat; i++)
    {
        const auto result = variant.execute(in, out, lut3D);
        sum._upload += result._upload;
        sum._execute += result._execute;
        sum._download += result._download;
    }

    return {
        ._upload = sum._upload / repeat,
        ._execute = sum._execute / repeat,
        ._download = sum._download / repeat,
    };
}

void compare_images(const uint8_t* imageA, const uint8_t* imageB, size_t width, size_t height, uint8_t maxDiff)
{
    const auto pixels = width * height;

    for(size_t i = 0; i < pixels; ++i)
    {
        const auto offset = i * 4 * sizeof(uint8_t);

        const auto aR = imageA[offset + 0];
        const auto aG = imageA[offset + 1];
        const auto aB = imageA[offset + 2];
        const auto aA = imageA[offset + 3];

        const auto bR = imageB[offset + 0];
        const auto bG = imageB[offset + 1];
        const auto bB = imageB[offset + 2];
        const auto bA = imageB[offset + 3];

        if(abs(aR - bR) > maxDiff)
            assert(false && abs(aR - bR));
        if(abs(aG - bG) > maxDiff)
            assert(false && abs(aR - bR));
        if(abs(aB - bB) > maxDiff)
            assert(false && abs(aR - bR));
        if(abs(aA - bA) > maxDiff)
            assert(false && abs(aR - bR));
    }
}

int run_tetrahedral_interpolation()
{
    const auto modules = {
        path(BASE_RELEASE "apply_lut3D.ptx"),
    };
    const auto image_in = "data/in/in16k.png";
    const auto image_out = "data/out/out16k.png";
    const auto lut_in = "data/lut/6-1a_BT709_HLG_Type3_Scene_UpMapping_nocomp-v1_5.cube";
    const auto ref_out = "data/ref/ref16k.png";

    ifstream file(lut_in);
    auto cube_opt = load_cube<float>(file);
    if(!cube_opt)
        return -1;
    auto& cube = *cube_opt;

    size_t width, height;
    auto image = load_image(image_in, width, height);
    if(!image)
        return -1;

    size_t ref_width, ref_height;
    auto ref_image = load_image(ref_out, ref_width, ref_height);
    if(!ref_image)
        return -1;

    assert(width == ref_width);
    assert(height == ref_height);

    auto output = std::make_unique<uint8_t[]>(width * height * 4 * sizeof(uint8_t));

    auto tI = tetrahedral_interpolation::tetrahedral_interpolation_cuda(modules);
    const auto lut3D = tI.create_lut3D(cube.data(), cube.size());

    constexpr size_t repeat = 100;

    for(auto& id : tI.variants1D())
    {
        cout << tI.variant_name(id) << endl;

        const auto result = calculate_avg_time_1D(tI, id, image.get(), output.get(), width * height, lut3D, repeat);

        cout << "\tUpload: " << duration_cast<milliseconds_double>(result._upload).count() << endl;
        cout << "\tExecute: " << duration_cast<milliseconds_double>(result._execute).count() << endl;
        cout << "\tDownload: " << duration_cast<milliseconds_double>(result._download).count() << endl;

        compare_images(output.get(), ref_image.get(), width, height, static_cast<uint8_t>(256.0f * 0.03f));
    }

    for(auto& id : tI.variants2D())
    {
        cout << tI.variant_name(id) << endl;

        const auto result = calculate_avg_time_2D(tI, id, image.get(), output.get(), width, height, lut3D, repeat);

        cout << "\tUpload: " << duration_cast<milliseconds_double>(result._upload).count() << endl;
        cout << "\tExecute: " << duration_cast<milliseconds_double>(result._execute).count() << endl;
        cout << "\tDownload: " << duration_cast<milliseconds_double>(result._download).count() << endl;

        compare_images(output.get(), ref_image.get(), width, height, static_cast<uint8_t>(256.0f * 0.03f));
    }

    return 0;
}

std::tuple<std::vector<float>, size_t> generate_image(size_t width, size_t height)
{
    std::random_device randomDevice{};
    std::mt19937_64 generator{randomDevice()};
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    std::vector<float> imageData(4 * width * height);
    std::generate(std::begin(imageData), std::end(imageData), [&](){
        return distribution(generator);
    });

    return std::make_tuple(imageData, 4 * sizeof(float) * width);
}

memory_layout_overhead::variant_result calculate_avg_time_2D(
    memory_layout_overhead::memory_layout_overhead& mLO,
    const variant_id& id,
    size_t width, size_t height, std::span<const float> in, size_t inPitchInBytes,
    std::span<uint16_t> out, size_t outPitchInBytes,
    size_t repeat = 100
)
{
    memory_layout_overhead::variant_result sum{};

    auto variant = mLO.create_variant2D(id, width, height, in, inPitchInBytes);

    for(size_t i = 0; i < repeat; i++)
    {
        const auto result = variant.execute(out, outPitchInBytes);
        sum._execute += result._execute;
        sum._download += result._download;
    }

    return {
        ._execute = sum._execute / repeat,
        ._download = sum._download / repeat,
    };
}

void compare_images(const uint8_t* imageA, const uint8_t* imageB,
                    size_t imageAPitchInBytes, size_t imageBPitchInBytes,
                    size_t width, size_t height, uint16_t maxDiff)
{
    auto floatTo16bit = [](float v)
    {
        constexpr uint16_t max16bitValue = 65535;
        constexpr float scale = max16bitValue;
        return static_cast<uint16_t>(std::round(v * scale));
    };

    for (size_t y = 0; y < height; ++y)
    {
        auto imageARow = imageA + imageAPitchInBytes * y;
        auto imageBRow = imageB + imageBPitchInBytes * y;

        for (size_t x = 0; x < width; ++x)
        {
            auto imageAPixel = reinterpret_cast<const float*>(imageARow + 4 * sizeof(float) * x);
            auto imageBPixel = reinterpret_cast<const uint16_t*>(imageBRow + 4 * sizeof(uint16_t) * x);

            const auto aR = floatTo16bit(imageAPixel[0]);
            const auto aG = floatTo16bit(imageAPixel[1]);
            const auto aB = floatTo16bit(imageAPixel[2]);
            const auto aA = floatTo16bit(imageAPixel[3]);

            const auto bR = imageBPixel[0];
            const auto bG = imageBPixel[1];
            const auto bB = imageBPixel[2];
            const auto bA = imageBPixel[3];

            if(abs(aR - bR) > maxDiff)
                assert(false && abs(aR - bR));
            if(abs(aG - bG) > maxDiff)
                assert(false && abs(aR - bR));
            if(abs(aB - bB) > maxDiff)
                assert(false && abs(aR - bR));
            if(abs(aA - bA) > maxDiff)
                assert(false && abs(aR - bR));
        }
    }
}

int run_memory_layout_overhead()
{
    const auto modules = {
        path(BASE_RELEASE "texture_readback.ptx"),
    };

    constexpr size_t width = 3840;
    constexpr size_t height = 2160;

    const auto [in, inPitchInBytes] = generate_image(width, height);
    auto out = std::vector<uint16_t>(4 * width * height);
    const auto outPitchInBytes = 4 * sizeof(uint16_t) * width;

    auto output = std::make_unique<uint8_t[]>(width * height * 4 * sizeof(uint8_t));

    auto mLO = memory_layout_overhead::memory_layout_overhead_cuda(modules);

    constexpr size_t repeat = 1;

    for(auto& id : mLO.variants2D())
    {
        cout << mLO.variant_name(id) << endl;

        const auto result = calculate_avg_time_2D(mLO, id, width, height, in, inPitchInBytes, out, outPitchInBytes, repeat);

        cout << "\tExecute: " << duration_cast<milliseconds_double>(result._execute).count() << endl;
        cout << "\tDownload: " << duration_cast<milliseconds_double>(result._download).count() << endl;

        compare_images(
            reinterpret_cast<const uint8_t*>(in.data()),
            reinterpret_cast<const uint8_t*>(out.data()),
            inPitchInBytes, outPitchInBytes,
            width, height, static_cast<uint8_t>(65535 * 0.004f)
        );
    }

    return 0;
}

int main(int argc, const char* const argv[])
{
    return run_memory_layout_overhead();
}