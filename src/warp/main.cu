#include <cuda.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#define ASSERT(exp)                                                  \
  do {                                                               \
    if (!(exp)) {                                                    \
      fprintf(stderr, "[%s:%d]" #exp " fail\n", __FILE__, __LINE__); \
      exit(1);                                                       \
    }                                                                \
  } while (0)

#define CUDA_CHECK(exp)                                                                                     \
  do {                                                                                                      \
    cudaError_t err = (exp);                                                                                \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d]" #exp " got CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

#define KERNEL_CHECK(msg)                                                                                   \
  do {                                                                                                      \
    cudaError_t err = cudaGetLastError();                                                                   \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d] %s got CUDA error: %s\n", __FILE__, __LINE__, msg, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

__global__ void Kernel(int* thread_arr, int* warp_arr, int* lane_arr, int* block_arr, int size) {
  int dim = blockDim.x * gridDim.x;
  assert(dim == size);
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;
  int warp_idx = threadIdx.x / warpSize;
  int lane_idx;

  asm volatile("mov.u32  %0,  %%laneid;" : "=r"(lane_idx));
  thread_arr[i] = thread_idx;
  block_arr[i] = block_idx;
  warp_arr[i] = warp_idx;
  lane_arr[i] = lane_idx;
}

struct Test {
  int* d_thread_arr;
  int* d_warp_arr;
  int* d_lane_arr;
  int* d_block_arr;
  int* h_thread_arr;
  int* h_warp_arr;
  int* h_lane_arr;
  int* h_block_arr;
  int grid_dim;
  int block_dim;
  int warp_size;
  cudaDeviceProp prop;
  cudaStream_t stream;

  __host__ Test() : grid_dim(2), block_dim(64) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));

    warp_size = prop.warpSize;
    const int size = grid_dim * block_dim;
    CUDA_CHECK(cudaMalloc(&d_thread_arr, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc(&d_warp_arr, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc(&d_lane_arr, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc(&d_block_arr, sizeof(int) * size));
    h_thread_arr = static_cast<int*>(malloc(sizeof(int) * size));
    ASSERT(h_thread_arr);
    h_warp_arr = static_cast<int*>(malloc(sizeof(int) * size));
    ASSERT(h_warp_arr);
    h_lane_arr = static_cast<int*>(malloc(sizeof(int) * size));
    ASSERT(h_lane_arr);
    h_block_arr = static_cast<int*>(malloc(sizeof(int) * size));
    ASSERT(h_block_arr);
  }

  __host__ ~Test() {
    free(h_thread_arr);
    free(h_warp_arr);
    free(h_lane_arr);
    free(h_block_arr);
    CUDA_CHECK(cudaFree(d_thread_arr));
    CUDA_CHECK(cudaFree(d_warp_arr));
    CUDA_CHECK(cudaFree(d_lane_arr));
    CUDA_CHECK(cudaFree(d_block_arr));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Print(const std::string& msg, int* x, int size) {
    std::cout << msg;
    for (int i = 0; i < size - 1; ++i) std::cout << x[i] << ",";
    std::cout << x[size - 1] << std::endl;
  }

  __host__ void Run() {
    int grid_dim = 2;
    int block_dim = 64;  // 2 warp
    int size = grid_dim * block_dim;
    Kernel<<<grid_dim, block_dim, 0, stream>>>(d_thread_arr, d_warp_arr, d_lane_arr, d_block_arr, size);
    KERNEL_CHECK("Launch Kernel");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_thread_arr, d_thread_arr, sizeof(int) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_warp_arr, d_warp_arr, sizeof(int) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lane_arr, d_lane_arr, sizeof(int) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_arr, d_block_arr, sizeof(int) * size, cudaMemcpyDeviceToHost));
    Print("thread: ", h_thread_arr, size);
    Print("warp: ", h_warp_arr, size);
    Print("lane: ", h_lane_arr, size);
    Print("block: ", h_block_arr, size);
  }
};

int main(int argc, char* argv[]) {
  auto test = Test();
  test.Run();
}
