#include <cuda.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

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

__global__ void Init(int* A_d, const size_t size) {
  assert(size == warpSize);
  A_d[threadIdx.x] = threadIdx.x;
}

__global__ void WarpDirectAccess(int* A, int* B) {
  // __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
  // mask: Which threads participate (usually 0xffffffff for all 32)
  // var: Variable to read from source thread
  // srcLane: Which thread to read from (0-31)
  // width: Warp subdivision size (default 32)
  auto pos = warpSize - 1;
  // thread[pos] broadcast its value to other threads
  auto value = __shfl_sync(0xffffffff, A[pos], pos);
  assert(value == A[pos]);
  B[threadIdx.x] = value;
}

__global__ void WarpReduce(int* A, int* out) {
  // Sum all array's elements
  int x = A[threadIdx.x];
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  if (threadIdx.x % warpSize == 0) {
    *out = x;
  }
}

struct WarpTest {
  int device_id = 0;
  int warp_size;
  cudaDeviceProp prop;
  int* A_h = nullptr;
  int* B_h = nullptr;
  int* A_d = nullptr;
  int* B_d = nullptr;
  int* out_d = nullptr;
  cudaStream_t stream;

  __host__ WarpTest() {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    warp_size = prop.warpSize;
    printf("warp size = %d\n", warp_size);
    A_h = static_cast<int*>(malloc(sizeof(int) * warp_size));
    ASSERT(!!A_h);
    B_h = static_cast<int*>(malloc(sizeof(int) * warp_size));
    ASSERT(!!B_h);
    CUDA_CHECK(cudaMalloc(&A_d, sizeof(int) * warp_size));
    CUDA_CHECK(cudaMalloc(&B_d, sizeof(int) * warp_size));
    CUDA_CHECK(cudaMalloc(&out_d, sizeof(int)));
  }

  __host__ ~WarpTest() {
    free(A_h);
    free(B_h);
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(out_d));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ __forceinline__ void VerifyWarpDirectAccess() {
    CUDA_CHECK(cudaMemcpy(A_h, A_d, sizeof(int) * warp_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_h, B_d, sizeof(int) * warp_size, cudaMemcpyDeviceToHost));
    auto pos = warp_size - 1;
    for (int i = 0; i < warp_size; ++i) ASSERT(A_h[pos] == B_h[i]);
  }

  __host__ __forceinline__ void VerifyWarpReduce() {
    int out_h = 0;
    const int ans = (warp_size - 1) * warp_size / 2;
    CUDA_CHECK(cudaMemcpy(&out_h, out_d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("WarpReduce result: %d\n", out_h);
    ASSERT(out_h == ans);
  }

  __host__ void Run() {
    // Test direct access
    Init<<<1, warp_size, 0, stream>>>(A_d, warp_size);
    KERNEL_CHECK("run A_d initialization");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    WarpDirectAccess<<<1, warp_size, 0, stream>>>(A_d, B_d);
    KERNEL_CHECK("run WarpDirectAccess");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    VerifyWarpDirectAccess();

    // Test reduce
    Init<<<1, warp_size, 0, stream>>>(A_d, warp_size);
    KERNEL_CHECK("run A_d initialization");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    WarpReduce<<<1, warp_size, 0, stream>>>(A_d, out_d);
    KERNEL_CHECK("run WarpReduce");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    VerifyWarpReduce();
  }
};

int main(int argc, char* argv[]) {
  auto warp = WarpTest();
  warp.Run();
}
