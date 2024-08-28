#pragma once

#include <tetrahedral_interpolation_core.hpp>

namespace tetrahedral_interpolation
{

class tetrahedral_interpolation_cuda_impl;

class tetrahedral_interpolation_cuda
    : public tetrahedral_interpolation
{
private:
    std::shared_ptr<tetrahedral_interpolation_cuda_impl> _impl;
public:
    tetrahedral_interpolation_cuda(const std::span<const std::filesystem::path> &modules);
public:
    ~tetrahedral_interpolation_cuda() override = default;
public:
    [[maybe_unused]] [[nodiscard]] std::vector<variant_id> variants1D() const override;
    [[maybe_unused]] [[nodiscard]] std::vector<variant_id> variants2D() const override;

    [[maybe_unused]] [[nodiscard]] std::string_view variant_name(const variant_id &id) const override;
protected:
    std::shared_ptr<lut3D_impl> create_lut3D_impl(const void *data, size_t lutSize) override;

    std::shared_ptr<variant1D_impl> create_variant1D_impl(const variant_id &id, size_t size) override;
    std::shared_ptr<variant2D_impl> create_variant2D_impl(const variant_id &id, size_t width, size_t height) override;
};

}
