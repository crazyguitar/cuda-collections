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

constexpr int BUFSIZE = 512;

__global__ void Kernel(int* x, int* y, int* out, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    out[idx] = x[idx] + y[idx];
  }
}

__host__ void Rand(int* arr, const int size, const int low, const int high) {
  std::random_device d;
  std::mt19937 g(d());
  std::uniform_int_distribution<int> dist(low, high);
  for (int i = 0; i < size; ++i) arr[i] = dist(g);
}

template <typename T>
struct Test {
  constexpr static int iterations = 1024;
  constexpr static int warmup = 64;

  int* host_input_x;
  int* host_input_y;
  int* host_output;
  int* dev_input_x;
  int* dev_input_y;
  int* dev_output;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaStream_t stream;

  __host__ Test() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  __host__ virtual ~Test() {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
  }

  __host__ void Verify(int* x, int* y, int* out, int size) {
    for (int i = 0; i < size; ++i) {
      assert(out[i] == (x[i] + y[i]));
    }
  }

  __host__ void Run() {
    float elapse;
    Rand(host_input_x, BUFSIZE, 0, 10);
    Rand(host_input_y, BUFSIZE, 0, 10);
    // warmup
    for (int i = 0; i < warmup; ++i) static_cast<T*>(this)->Launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) static_cast<T*>(this)->Launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapse, start, stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Verify(host_input_x, host_input_y, host_output, BUFSIZE);
    auto latency = elapse / iterations;
    printf("elapse: %f, latency: %f\n", elapse, latency);
  }
};

struct TestZeroCopy : public Test<TestZeroCopy> {
  __host__ __forceinline__ void Launch() {
    int grid_dim = 1;
    int block_dim = BUFSIZE;
    cudaLaunchConfig_t cfg{0};
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, Kernel, dev_input_x, dev_input_y, dev_output, BUFSIZE);
  }
};

struct TestCopy : public Test<TestCopy> {
  __host__ __forceinline__ void Launch() {
    CUDA_CHECK(cudaMemcpyAsync(dev_input_x, host_input_x, sizeof(int) * BUFSIZE, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_input_y, host_input_y, sizeof(int) * BUFSIZE, cudaMemcpyHostToDevice, stream));
    int grid_dim = 1;
    int block_dim = BUFSIZE;
    cudaLaunchConfig_t cfg{0};
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, Kernel, dev_input_x, dev_input_y, dev_output, BUFSIZE);
    CUDA_CHECK(cudaMemcpyAsync(host_output, dev_output, sizeof(int) * BUFSIZE, cudaMemcpyDeviceToHost, stream));
  }
};

struct TestHostRegister : public TestZeroCopy {
  __host__ TestHostRegister() : TestZeroCopy() {
    host_input_x = static_cast<int*>(malloc(sizeof(int) * BUFSIZE));
    host_input_y = static_cast<int*>(malloc(sizeof(int) * BUFSIZE));
    host_output = static_cast<int*>(malloc(sizeof(int) * BUFSIZE));
    CUDA_CHECK(cudaHostRegister(host_input_x, sizeof(int) * BUFSIZE, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(host_input_y, sizeof(int) * BUFSIZE, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(host_output, sizeof(int) * BUFSIZE, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_x, host_input_x, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_y, host_input_y, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_output, host_output, 0));
  }

  __host__ ~TestHostRegister() override {
    CUDA_CHECK(cudaHostUnregister(host_input_x));
    CUDA_CHECK(cudaHostUnregister(host_input_y));
    CUDA_CHECK(cudaHostUnregister(host_output));
    free(host_input_x);
    free(host_input_y);
    free(host_output);
  }
};

struct TestMallocHost : public TestZeroCopy {
  __host__ TestMallocHost() : TestZeroCopy() {
    CUDA_CHECK(cudaMallocHost(&host_input_x, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocHost(&host_input_y, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocHost(&host_output, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_x, host_input_x, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_y, host_input_y, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_output, host_output, 0));
  }

  __host__ ~TestMallocHost() override {
    CUDA_CHECK(cudaFreeHost(host_input_x));
    CUDA_CHECK(cudaFreeHost(host_input_y));
    CUDA_CHECK(cudaFreeHost(host_output));
  }
};

struct TestHostAlloc : public TestZeroCopy {
  __host__ TestHostAlloc() : TestZeroCopy() {
    CUDA_CHECK(cudaHostAlloc(&host_input_x, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&host_input_y, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&host_output, sizeof(int) * BUFSIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_x, host_input_x, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_y, host_input_y, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_output, host_output, 0));
  }

  __host__ ~TestHostAlloc() override {
    CUDA_CHECK(cudaFreeHost(host_input_x));
    CUDA_CHECK(cudaFreeHost(host_input_y));
    CUDA_CHECK(cudaFreeHost(host_output));
  }
};

struct TestMemcpy : public TestCopy {
  __host__ TestMemcpy() : TestCopy() {
    host_input_x = static_cast<int*>(malloc(sizeof(int) * BUFSIZE));
    host_input_y = static_cast<int*>(malloc(sizeof(int) * BUFSIZE));
    host_output = static_cast<int*>(malloc(sizeof(int) * BUFSIZE));
    CUDA_CHECK(cudaMalloc(&dev_input_x, sizeof(int) * BUFSIZE));
    CUDA_CHECK(cudaMalloc(&dev_input_y, sizeof(int) * BUFSIZE));
    CUDA_CHECK(cudaMalloc(&dev_output, sizeof(int) * BUFSIZE));
  }

  __host__ ~TestMemcpy() {
    CUDA_CHECK(cudaFree(dev_input_x));
    CUDA_CHECK(cudaFree(dev_input_y));
    CUDA_CHECK(cudaFree(dev_output));
    free(host_input_x);
    free(host_input_y);
    free(host_output);
  }
};

template <typename... T>
void Run() {
  (T{}.Run(), ...);
}

int main(int argc, char* argv[]) { Run<TestHostRegister, TestMallocHost, TestHostAlloc, TestMemcpy>(); }
