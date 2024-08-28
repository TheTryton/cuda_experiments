#pragma once

#include <cuda.h>
#include <exception>
#include <iostream>

static void check_error(CUresult error)
{
    if (error != CUDA_SUCCESS)
    {
        const char *errorString;
        cuGetErrorName(error, &errorString);
        throw std::exception(errorString);
    }
}

static CUfunction find_in_modules(const std::span<CUmodule> &modules, const char *name)
{
    CUfunction function;
    CUresult error;
    for (auto &module: modules)
    {
        error = cuModuleGetFunction(&function, module, name);
        if (error == CUDA_SUCCESS)
            return function;
    }

    check_error(error);
    return {};
}
