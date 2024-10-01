#include <cstdint>
#include <vector>
#include <chrono>
#include <cassert>
#include <iostream>
#include <format>

using namespace std;

double measure_ram_ram_transfer_speed(const vector<uint8_t> & src, vector<uint8_t> & dst, size_t iterations)
{
    assert(src.size() == dst.size());

    using seconds = chrono::duration<double, ratio<1, 1>>;

    const auto start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i)
    {
        memcpy(data(dst), data(src), sizeof(uint8_t) * size(src));
    }
    const auto stop = chrono::high_resolution_clock::now();

    const auto bytesTransferred = sizeof(uint8_t) * size(src);
    const auto timeTakenSeconds = chrono::duration_cast<seconds>(stop - start).count();

    const auto bytesPerSecond = static_cast<double>(bytesTransferred) / timeTakenSeconds * static_cast<double>(iterations);
    return bytesPerSecond;
}

double measure_gpuram_gpuram_transfer_speed(const vector<uint8_t> & src,size_t iterations)
{
    assert(src.size() == dst.size());

    uint8_t * srcDevice;
    cudaMalloc(&srcDevice, sizeof(uint8_t) * size(src));
    uint8_t * dstDevice;
    cudaMalloc(&dstDevice, sizeof(uint8_t) * size(src));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i = 0; i < iterations; ++i)
    {
        cudaMemcpyAsync(dstDevice, srcDevice, sizeof(uint8_t) * size(src), cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);

    cudaStreamSynchronize(0);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const auto bytesTransferred = sizeof(uint8_t) * size(src);
    const auto timeTakenSeconds = static_cast<double>(milliseconds) / 1000;

    const auto bytesPerSecond = static_cast<double>(bytesTransferred) / timeTakenSeconds * static_cast<double>(iterations);
    return bytesPerSecond;
}

double measure_ram_gpuram_transfer_speed(const vector<uint8_t> & src, size_t iterations)
{
    assert(src.size() == dst.size());

    uint8_t * srcHost;
    cudaHostAlloc(&srcHost, sizeof(uint8_t) * size(src), cudaHostAllocWriteCombined);
    std::memcpy(srcHost, data(src), sizeof(uint8_t) * size(src));
    uint8_t * dstDevice;
    cudaMalloc(&dstDevice, sizeof(uint8_t) * size(src));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i = 0; i < iterations; ++i)
    {
        cudaMemcpyAsync(dstDevice, srcHost, sizeof(uint8_t) * size(src), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);

    cudaStreamSynchronize(0);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const auto bytesTransferred = sizeof(uint8_t) * size(src);
    const auto timeTakenSeconds = static_cast<double>(milliseconds) / 1000;

    const auto bytesPerSecond = static_cast<double>(bytesTransferred) / timeTakenSeconds * static_cast<double>(iterations);
    return bytesPerSecond;
}

double measure_gpuram_ram_transfer_speed(const vector<uint8_t> & src, size_t iterations)
{
    assert(src.size() == dst.size());

    uint8_t * srcDevice;
    cudaMalloc(&srcDevice, sizeof(uint8_t) * size(src));
    uint8_t * dstHost;
    cudaHostAlloc(&dstHost, sizeof(uint8_t) * size(src), cudaHostAllocDefault);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i = 0; i < iterations; ++i)
    {
        cudaMemcpyAsync(srcDevice, dstHost, sizeof(uint8_t) * size(src), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);

    cudaStreamSynchronize(0);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const auto bytesTransferred = sizeof(uint8_t) * size(src);
    const auto timeTakenSeconds = static_cast<double>(milliseconds) / 1000;

    const auto bytesPerSecond = static_cast<double>(bytesTransferred) / timeTakenSeconds * static_cast<double>(iterations);
    return bytesPerSecond;
}

int main()
{
    constexpr size_t bytesCount = 4ull * 1024 * 1024 * 1024;
    constexpr size_t iterationCount = 10;

    vector<uint8_t> src(bytesCount);
    vector<uint8_t> dst(bytesCount);

    const auto transferSpeedRamRam = measure_ram_ram_transfer_speed(src, dst, iterationCount);
    const auto transferSpeedGpuRamGpuRam = measure_gpuram_gpuram_transfer_speed(src, iterationCount);
    const auto transferSpeedRamGpuRam = measure_gpuram_ram_transfer_speed(src, iterationCount);
    const auto transferSpeedGpuRamRam = measure_ram_gpuram_transfer_speed(src, iterationCount);

    cout << format("[CPU->CPU] Transfer speed {} GB/s", transferSpeedRamRam / 1024 / 1024 / 1024) << endl;
    cout << format("[GPU->GPU] Transfer speed {} GB/s", transferSpeedGpuRamGpuRam / 1024 / 1024 / 1024) << endl;
    cout << format("[CPU->GPU] Transfer speed {} GB/s", transferSpeedRamGpuRam / 1024 / 1024 / 1024) << endl;
    cout << format("[GPU->CPU] Transfer speed {} GB/s", transferSpeedGpuRamRam / 1024 / 1024 / 1024) << endl;

    return 0;
}

