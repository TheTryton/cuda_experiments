#include <tetrahedral_interpolation_core.hpp>

namespace tetrahedral_interpolation
{

lut3D::lut3D(const std::shared_ptr<lut3D_impl> &impl)
    : _impl(impl)
{
}

const lut3D_impl &lut3D::impl() const noexcept
{
    return *_impl;
}

variant1D::variant1D(const std::shared_ptr<variant1D_impl> &impl)
    : _impl(impl)
{
}

variant_result variant1D::execute(const void *in, void *out, const lut3D &lut3D)
{
    return _impl->execute(in, out, lut3D.impl());
}

variant2D::variant2D(const std::shared_ptr<variant2D_impl> &impl)
    : _impl(impl)
{
}

variant_result variant2D::execute(const void *in, void *out, const lut3D &lut3D)
{
    return _impl->execute(in, out, lut3D.impl());
}

lut3D tetrahedral_interpolation::create_lut3D(const void *data, size_t lutSize)
{
    return lut3D{create_lut3D_impl(data, lutSize)};
}

variant1D tetrahedral_interpolation::create_variant1D(const variant_id &id, size_t size)
{
    return variant1D{create_variant1D_impl(id, size)};
}

variant2D tetrahedral_interpolation::create_variant2D(const variant_id &id, size_t width, size_t height)
{
    return variant2D{create_variant2D_impl(id, width, height)};
}

}
