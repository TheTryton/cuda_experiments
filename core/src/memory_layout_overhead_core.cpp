#include <memory_layout_overhead_core.hpp>

namespace memory_layout_overhead
{

variant2D::variant2D(const std::shared_ptr<variant2D_impl> &impl)
    : _impl(impl)
{
}

variant_result variant2D::execute(std::span<uint16_t> out, size_t outPitchInBytes)
{
    return _impl->execute(out, outPitchInBytes);
}

variant2D memory_layout_overhead::create_variant2D(const variant_id &id, size_t width, size_t height, std::span<const float> src, size_t srcPitchInBytes)
{
    return variant2D{create_variant2D_impl(id, width, height, src, srcPitchInBytes)};
}

}
