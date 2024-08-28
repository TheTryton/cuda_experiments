#pragma once

#include <variant.hpp>

namespace tetrahedral_interpolation
{

struct variant_result
{
    std::chrono::nanoseconds _upload;
    std::chrono::nanoseconds _execute;
    std::chrono::nanoseconds _download;
};

class lut3D_impl
{
public:
    virtual ~lut3D_impl() = default;
};

class lut3D
{
    friend class variant1D;
    friend class variant2D;
    friend class tetrahedral_interpolation;
private:
    std::shared_ptr<lut3D_impl> _impl;
private:
    explicit lut3D(const std::shared_ptr<lut3D_impl>& impl);
private:
    [[nodiscard]] const lut3D_impl& impl() const noexcept;
};

class variant1D_impl
{
public:
    virtual ~variant1D_impl() = default;
public:
    virtual variant_result execute(const void* in, void* out, const lut3D_impl& lut3D) = 0;
};

class variant2D_impl
{
public:
    virtual ~variant2D_impl() = default;
public:
    virtual variant_result execute(const void* in, void* out, const lut3D_impl& lut3D) = 0;
};

class variant1D
{
    friend class tetrahedral_interpolation;
private:
    std::shared_ptr<variant1D_impl> _impl;
private:
    explicit variant1D(const std::shared_ptr<variant1D_impl>& impl);
public:
    variant_result execute(const void* in, void* out, const lut3D& lut3D);
};

class variant2D
{
    friend class tetrahedral_interpolation;
private:
    std::shared_ptr<variant2D_impl> _impl;
private:
    explicit variant2D(const std::shared_ptr<variant2D_impl>& impl);
public:
    variant_result execute(const void* in, void* out, const lut3D& lut3D);
};

class tetrahedral_interpolation
{
public:
    virtual ~tetrahedral_interpolation() = default;
public:
    [[maybe_unused]] [[nodiscard]] virtual std::vector<variant_id> variants1D() const = 0;
    [[maybe_unused]] [[nodiscard]] virtual std::vector<variant_id> variants2D() const = 0;

    [[maybe_unused]] [[nodiscard]] virtual std::string_view variant_name(const variant_id& id) const = 0;

    lut3D create_lut3D(const void* data, size_t lutSize);
    variant1D create_variant1D(const variant_id& id, size_t size);
    variant2D create_variant2D(const variant_id& id, size_t width, size_t height);
protected:
    virtual std::shared_ptr<lut3D_impl> create_lut3D_impl(const void* data, size_t lutSize) = 0;

    virtual std::shared_ptr<variant1D_impl> create_variant1D_impl(const variant_id& id, size_t size) = 0;
    virtual std::shared_ptr<variant2D_impl> create_variant2D_impl(const variant_id& id, size_t width, size_t height) = 0;
};

}
