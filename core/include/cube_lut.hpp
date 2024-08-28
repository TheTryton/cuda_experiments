#pragma once

#include <memory>
#include <tuple>
#include <istream>
#include <string>
#include <string_view>
#include <optional>
#include <charconv>
#include <sstream>

template<typename ValueType>
class cube_lut
{
private:
    std::unique_ptr<ValueType[]> _cube;
    size_t _size;
public:
    cube_lut(std::unique_ptr<ValueType[]>&& cube, size_t size)
        : _cube(std::move(cube))
        , _size(size)
    { }
public:
    const ValueType* data() const noexcept
    {
        return _cube.get();
    }
    ValueType* data() noexcept
    {
        return _cube.get();
    }
public:
    [[nodiscard]] size_t size() const noexcept
    {
        return _size;
    }
};

template<typename ValueType>
std::optional<size_t> load_cube_size(std::istream& istream)
{
    constexpr std::string_view lut3d_size = "LUT_3D_SIZE";

    std::string in;

    while (std::getline(istream, in))
    {
        std::string_view line = in;

        auto sizePos = line.find(lut3d_size);

        if (sizePos != std::string_view::npos)
        {
            auto lut3d_size_str = line.substr(sizePos + size(lut3d_size) + 1);

            size_t lut3d_size_value;
            auto result = std::from_chars(
                lut3d_size_str.data(),
                lut3d_size_str.data() + lut3d_size_str.size(),
                lut3d_size_value
            );

            if (result.ec != std::errc{})
                return std::nullopt;

            return lut3d_size_value;
        }
    }

    return std::nullopt;
}

template<typename ValueType>
std::optional<cube_lut<ValueType>> load_cube(std::istream& istream)
{
    auto loaded_size = load_cube_size<ValueType>(istream);
    if(!loaded_size)
        return std::nullopt;

    auto size = *loaded_size;
    auto data = std::make_unique<ValueType[]>(size * size * size * 4);

    size_t index = 0;
    std::string line;
    while (std::getline(istream, line))
    {
        if(line.empty() || !std::isdigit(line[0]))
            continue;

        auto ss = std::istringstream (line);
        ValueType x, y, z;
        ss >> x >> y >> z;
        data[index * 4 + 0] = x;
        data[index * 4 + 1] = y;
        data[index * 4 + 2] = z;
        data[index * 4 + 3] = static_cast<ValueType>(0);
        ++index;
    }

    if(index != size * size * size)
        return std::nullopt;

    return cube_lut<ValueType>(std::move(data), size);
}
