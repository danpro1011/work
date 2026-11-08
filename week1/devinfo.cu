// devinfo.cu - print the GPU numbers you'll compare every kernel against.
// Build: nvcc -O3 -arch=sm_75 devinfo.cu -o devinfo && ./devinfo
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    // CUDA 13 removed clockRate/memoryClockRate from cudaDeviceProp, so use attributes.
    int clk_khz = 0, memclk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);

    // DDR: 2 transfers per memory clock.
    double bw_gbs = 2.0 * memclk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    // Turing: 64 FP32 cores per SM, 2 FLOPs (FMA) per core per cycle.
    double fp32_tflops = (double)p.multiProcessorCount * 64 * 2 * clk_khz * 1e3 / 1e12;

    printf("%s | SMs: %d | sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    printf("Peak bandwidth ~ %.0f GB/s | Peak FP32 ~ %.1f TFLOP/s (at %.0f MHz)\n",
           bw_gbs, fp32_tflops, clk_khz / 1e3);
    printf("Global mem: %.1f GB | Shared mem/block: %zu KB | L2: %d KB\n",
           p.totalGlobalMem / 1e9, p.sharedMemPerBlock / 1024, p.l2CacheSize / 1024);
    return 0;
}
