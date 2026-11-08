// matmul.cu - benchmark harness: cuBLAS baseline + your kernels.
// Build: nvcc -O3 -arch=sm_75 matmul.cu -o matmul -lcublas && ./matmul
//
// To add a kernel: write a __global__ function and a launch_xxx(...) wrapper,
// then add one line to the `kernels` list in main().
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

typedef void (*Launcher)(const float*, const float*, float*, int, int, int);

// ---------------- kernels (row-major: A is MxK, B is KxN, C is MxN) ----------------

// Naive: one thread per output element.
// NOTE: x indexes ROWS on purpose, so neighboring threads read A and write C with a
// large stride (uncoalesced). Stretch goal: swap row/col so x indexes columns and
// measure what coalescing alone buys you.
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16), grid((M + 15) / 16, (N + 15) / 16);
    matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
}

// TODO (you write this): matmul_tiled + launch_tiled.
//   - shared-memory tiles (16x16 or 32x32)
//   - two __syncthreads(): after loading the tile, and after computing with it
//   - handle M, N, K that are not multiples of the tile (size 1000 tests this)
// Then add {"tiled", launch_tiled} to the kernels list below.

// ---------------- cuBLAS baseline ----------------
cublasHandle_t g_handle;
void launch_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    const float alpha = 1.f, beta = 0.f;
    // cuBLAS is column-major, so compute C^T = B^T * A^T to get row-major C.
    cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// ---------------- harness ----------------
// Returns the median time in ms over `iters` runs after `warmup` untimed runs.
float time_kernel(Launcher f, const float* A, const float* B, float* C, int M, int N, int K,
                  int warmup = 3, int iters = 20) {
    for (int i = 0; i < warmup; i++) f(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    std::vector<float> ts;
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(s);
        f(A, B, C, M, N, K);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        ts.push_back(ms);
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

struct Entry { const char* name; Launcher fn; };

int main() {
    cublasCreate(&g_handle);
    // cuBLAS must be first: it is the correctness reference and the % baseline.
    std::vector<Entry> kernels = {
        {"cublas", launch_cublas},
        {"naive",  launch_naive},
        // {"tiled", launch_tiled},   // <- add yours here
    };
    int sizes[] = {512, 1000, 1024, 2048, 4096};   // 1000 is not a tile multiple: tests edge handling

    printf("%-8s %5s %10s %10s %8s %s\n", "kernel", "size", "ms", "GFLOP/s", "%cuBLAS", "max_err");
    for (int n : sizes) {
        int M = n, N = n, K = n;
        size_t elems = (size_t)n * n, bytes = elems * sizeof(float);
        std::vector<float> hA(elems), hB(elems), hRef(elems), hC(elems);
        for (auto& x : hA) x = (float)rand() / RAND_MAX * 2 - 1;
        for (auto& x : hB) x = (float)rand() / RAND_MAX * 2 - 1;

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        double flops = 2.0 * M * N * K, base_gflops = 0;
        for (size_t i = 0; i < kernels.size(); i++) {
            // Correctness first: run once, compare against cuBLAS output.
            CUDA_CHECK(cudaMemset(dC, 0, bytes));
            kernels[i].fn(dA, dB, dC, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
            if (i == 0) hRef = hC;
            float err = 0;
            for (size_t j = 0; j < elems; j++) err = fmaxf(err, fabsf(hC[j] - hRef[j]));

            float ms = time_kernel(kernels[i].fn, dA, dB, dC, M, N, K);
            double gflops = flops / (ms * 1e-3) / 1e9;
            if (i == 0) base_gflops = gflops;
            printf("%-8s %5d %10.3f %10.1f %7.1f%% %.2e %s\n", kernels[i].name, n, ms, gflops,
                   100.0 * gflops / base_gflops, err, err > 1e-2 ? "<-- WRONG" : "");
        }
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
    cublasDestroy(g_handle);
    return 0;
}
