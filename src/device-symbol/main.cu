#include <cuda.h>

#include <cassert>
#include <cstdio>
#include <random>

#define CUDA_CHECK(exp)                                                                                     \
  do {                                                                                                      \
    cudaError_t err = (exp);                                                                                \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d]" #exp " got CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

constexpr int BUFSIZE = 16;

// define a global device symbol. The device symbol is similar to global variable
// but it can only be used by device
__device__ int const_dev_data[BUFSIZE];

__global__ void Kernel(int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    ++const_dev_data[idx];
  }
}

__host__ void Rand(int* arr, const int size, const int low, const int high) {
  std::random_device d;
  std::mt19937 g(d());
  std::uniform_int_distribution<int> dist(low, high);
  for (int i = 0; i < size; ++i) arr[i] = dist(g);
}

struct Test {
  int input[BUFSIZE] = {0};
  cudaStream_t stream;

  __host__ Test() {
    Rand(input, BUFSIZE, 0, 10);
    CUDA_CHECK(cudaStreamCreate(&stream));
    // constant symbol doesn't need to alloc device memory manually
    CUDA_CHECK(cudaMemcpyToSymbol(const_dev_data, input, sizeof(int) * BUFSIZE));
  }

  __host__ ~Test() {
    // __device__ variable cannot be directly read in a host function
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Verify(int* output, int size) {
    for (int i = 0; i < size; ++i) {
      assert(output[i] == (input[i] + 1));
    }
  }

  __host__ void Run() {
    int output[BUFSIZE] = {0};
    int grid_dim = 1;
    int block_dim = BUFSIZE;
    cudaLaunchConfig_t cfg{0};
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, Kernel, BUFSIZE);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyFromSymbol(output, const_dev_data, sizeof(int) * BUFSIZE));
    Verify(output, BUFSIZE);
  }
};

int main(int argc, char* argv[]) {
  Test test;
  test.Run();
}
