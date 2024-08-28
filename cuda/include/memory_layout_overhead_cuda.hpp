#pragma once

#include <memory_layout_overhead_core.hpp>

namespace memory_layout_overhead
{

class memory_layout_overhead_cuda_impl;

class memory_layout_overhead_cuda
    : public memory_layout_overhead
{
private:
    std::shared_ptr<memory_layout_overhead_cuda_impl> _impl;
public:
    memory_layout_overhead_cuda(const std::span<const std::filesystem::path> &modules);
public:
    ~memory_layout_overhead_cuda() override = default;
public:
    [[maybe_unused]] [[nodiscard]] std::vector<variant_id> variants2D() const override;

    [[maybe_unused]] [[nodiscard]] std::string_view variant_name(const variant_id &id) const override;
protected:
    std::shared_ptr<variant2D_impl> create_variant2D_impl(const variant_id &id, size_t width, size_t height, std::span<const float> in, size_t inPitchInBytes) override;
};

}