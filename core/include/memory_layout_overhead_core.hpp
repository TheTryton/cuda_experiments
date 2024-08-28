#pragma once

#include <variant.hpp>

namespace memory_layout_overhead
{

struct variant_result
{
    std::chrono::nanoseconds _execute;
    std::chrono::nanoseconds _download;
};

class variant2D_impl
{
public:
    virtual ~variant2D_impl() = default;
public:
    virtual variant_result execute(std::span<uint16_t> out, size_t outPitchInBytes) = 0;
};

class variant2D
{
    friend class memory_layout_overhead;
private:
    std::shared_ptr<variant2D_impl> _impl;
private:
    explicit variant2D(const std::shared_ptr<variant2D_impl>& impl);
public:
    variant_result execute(std::span<uint16_t> out, size_t outPitchInBytes);
};

class memory_layout_overhead
{
public:
    virtual ~memory_layout_overhead() = default;
public:
    [[maybe_unused]] [[nodiscard]] virtual std::vector<variant_id> variants2D() const = 0;

    [[maybe_unused]] [[nodiscard]] virtual std::string_view variant_name(const variant_id& id) const = 0;

    variant2D create_variant2D(const variant_id& id, size_t width, size_t height, std::span<const float> in, size_t inPitchInBytes);
protected:
    virtual std::shared_ptr<variant2D_impl> create_variant2D_impl(const variant_id& id, size_t width, size_t height, std::span<const float> in, size_t inPitchInBytes) = 0;
};

}

