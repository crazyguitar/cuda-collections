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

constexpr int BUFSIZE = 128;

__global__ void Kernel(int* data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    ++data[idx];
  }
}

__host__ void Rand(int* arr, const int size, const int low, const int high) {
  std::random_device d;
  std::mt19937 g(d());
  std::uniform_int_distribution<int> dist(low, high);
  for (int i = 0; i < size; ++i) arr[i] = dist(g);
}

struct Test {
  int input[BUFSIZE];
  int* host_ptr;
  int* dev_ptr;
  cudaStream_t stream;

  __host__ Test() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaHostAlloc(&host_ptr, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    Rand(host_ptr, BUFSIZE, 0, 10);
    memcpy(input, host_ptr, sizeof(int) * BUFSIZE);
  }

  __host__ ~Test() {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(host_ptr));
  }

  __host__ void Verify(int* input, int* output, int size) {
    for (int i = 0; i < size; ++i) {
      assert(output[i] == (input[i] + 1));
    }
  }

  __host__ void Run() {
    int grid_dim = 1;
    int block_dim = BUFSIZE;
    cudaLaunchConfig_t cfg{0};
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, Kernel, dev_ptr, BUFSIZE);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Verify(input, host_ptr, BUFSIZE);
  }
};

int main(int argc, char* argv[]) {
  Test test;
  test.Run();
}
