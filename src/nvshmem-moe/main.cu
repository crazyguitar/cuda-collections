#include <cuda.h>
#include <curand_kernel.h>
#include <errno.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <utility>
#include <vector>

#define CHECK(exp)                                                                            \
  do {                                                                                        \
    auto rc = (exp);                                                                          \
    if (rc < 0) {                                                                             \
      fprintf(stderr, "[%s:%d]" #exp "fail. err: %s\n", __FILE__, __LINE__, strerror(errno)); \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

#define CUDA_CHECK(exp)                                                                                     \
  do {                                                                                                      \
    cudaError_t err = (exp);                                                                                \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d]" #exp " got CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

struct NVSHMEM {
  constexpr static size_t BUFSIZE = 256;

  int mype;
  int npes;
  int mype_node;
  char hostname[BUFSIZE] = {0};

  NVSHMEM() {
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CHECK(gethostname(hostname, BUFSIZE));
    CUDA_CHECK(cudaSetDevice(mype_node));
  }

  ~NVSHMEM() { nvshmem_finalize(); }

  friend std::ostream& operator<<(std::ostream& os, const NVSHMEM& nvshmem) {
    const auto& hostname = nvshmem.hostname;
    const auto& mype = nvshmem.mype;
    const auto& npes = nvshmem.npes;
    const auto& mype_node = nvshmem.mype_node;
    return os << "[" << hostname << "] mype: " << mype << " npes: " << npes << " mype_node: " << mype_node;
  }
};

__device__ __forceinline__ void InitIndices(curandState& state, int* d_indices, int k, int tokens, int num_experts) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
    d_indices[i * k] = curand(&state) % num_experts;
#pragma unroll
    for (int j = 1; j < k; ++j) {
      d_indices[j + i * k] = (d_indices[i * k] + 1) % num_experts;
    }
  }
}

__device__ __forceinline__ void InitTokens(curandState& state, float* d_x, int tokens, int input_dim) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < input_dim; ++j) {
      d_x[j + i * input_dim] = curand_uniform(&state);
    }
  }
}

__device__ __forceinline__ void Count(int* d_expert_counts, int* d_indices, int tokens, int k, int num_experts, int mype) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < k; ++j) {
      int expert_idx = d_indices[j + i * k];
      atomicAdd(&d_expert_counts[expert_idx + num_experts * mype], 1);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void Dispatch(float* d_x) {}
__device__ __forceinline__ void Forward(float* d_x, float* d_y, int tokens, int input_dim, int output_dim) {
  // This function only coypies d_x to d_y w/o doing actually MLP forward.
  assert(input_dim == output_dim);
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < input_dim; ++j) {
      d_y[j + i * input_dim] = d_x[j + i * input_dim];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void Combine(float* d_y) {}

__global__ void MoEKernel(
    float* d_x,
    float* d_y,
    int* d_indices,
    int* d_expert_counts,
    int seed,
    int k,
    int tokens,
    int input_dim,
    int output_dim,
    int num_experts,
    int num_local_experts,
    int mype,
    int npes
) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(seed, idx, 0, &rand_state);
  InitIndices(rand_state, d_indices, k, tokens, num_experts);
  InitTokens(rand_state, d_x, tokens, input_dim);
  __syncthreads();

  Count(d_expert_counts, d_indices, tokens, k, num_experts, mype);
  Dispatch(d_x);
  Forward(d_x, d_y, tokens, input_dim, output_dim);
  Combine(d_y);
}

struct MoE {
  constexpr static int batch_size = 4;
  constexpr static int sequence_len = 1024;
  constexpr static int input_dim = 512;
  constexpr static int output_dim = 512;
  constexpr static int num_local_experts = 2;
  constexpr static int k = 2;
  constexpr static int seed = 123;
  NVSHMEM nvshmem;
  cudaDeviceProp prop;
  cudaStream_t stream;

  /* MoE attr */
  int tokens;
  int num_experts;
  int* d_indices;
  int* d_expert_counts;
  float* d_x;
  float* d_y;

  __host__ MoE() {
    auto npes = nvshmem.npes;
    tokens = batch_size * sequence_len;
    num_experts = num_local_experts * nvshmem.npes;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * tokens * k));
    d_x = static_cast<float*>(nvshmem_malloc(sizeof(float) * tokens * input_dim));
    d_y = static_cast<float*>(nvshmem_malloc(sizeof(float) * tokens * output_dim));
    d_expert_counts = static_cast<int*>(nvshmem_malloc(sizeof(int) * npes * num_experts));
  }

  __host__ ~MoE() {
    nvshmem_free(d_x);
    nvshmem_free(d_y);
    nvshmem_free(d_expert_counts);
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Run() {
    auto mype = nvshmem.mype;
    auto npes = nvshmem.npes;
    cudaLaunchConfig_t cfg{0};
    int block_dim = std::min(tokens, prop.maxThreadsPerBlock);
    int grid_dim = 1;  // use single sm
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(
        &cfg, MoEKernel, d_x, d_y, d_indices, d_expert_counts, seed, k, tokens, input_dim, output_dim, num_experts, num_local_experts, mype, npes
    );
  }
};

int main(int argc, char* argv[]) {
  auto moe = MoE();
  moe.Run();
}
