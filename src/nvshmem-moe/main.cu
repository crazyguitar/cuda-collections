#include <cuda.h>
#include <curand_kernel.h>
#include <errno.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
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

#define DEBUG(mype, ...) \
  do {                   \
    if (mype == 0) {     \
      __VA_ARGS__        \
    }                    \
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

struct Indexer {
  int thread_idx;
  int block_idx;
  int block_dim;
  int warp_idx;
  int warp_dim;
  int lane_idx;

  Indexer() = delete;
  __device__ Indexer(int thread_idx, int block_idx, int block_dim, int warp_dim)
      : thread_idx{thread_idx}, block_idx{block_idx}, block_dim{block_dim}, warp_dim{warp_dim} {
    warp_idx = thread_idx / warp_dim;
    asm volatile("mov.u32  %0,  %%laneid;" : "=r"(lane_idx));
  }
};

struct TokenIndexer : public Indexer {
  int global_token_idx;

  TokenIndexer() = delete;
  __device__ TokenIndexer(int thread_idx, int block_idx, int block_dim, int warp_dim) : Indexer{thread_idx, block_idx, block_dim, warp_dim} {
    global_token_idx = thread_idx + block_dim * block_idx;
  }
};

__device__ void InitIndices(curandState& state, const TokenIndexer& indexer, int* d_indices, const int num_experts) {
  auto token_idx = indexer.global_token_idx;
  d_indices[token_idx] = curand(&state) % num_experts;
}

__device__ void InitTokens(curandState& state, const TokenIndexer& indexer, float* d_x, const int tokens, const int input_dim) {
  auto token_idx = indexer.global_token_idx;
#pragma unroll
  for (int i = 0; i < input_dim; ++i) {
    int elem_idx = i + token_idx * input_dim;
    d_x[elem_idx] = curand_uniform(&state);
  }
}

// TODO
__device__ void Dispatch(float* d_x) {}

// TODO
__device__ void Combine(float* d_y) {}

__global__ void
Forward(float* d_x, float* d_y, int* d_indices, int seed, int tokens, int input_dim, int output_dim, int num_experts, int num_local_experts) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= tokens) return;
  const auto indexer = TokenIndexer(threadIdx.x, blockIdx.x, blockDim.x, warpSize);
  curandState rand_state;
  curand_init(seed, idx, 0, &rand_state);
  InitIndices(rand_state, indexer, d_indices, num_experts);
  InitTokens(rand_state, indexer, d_x, tokens, input_dim);
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
  float* d_x;
  float* d_y;

  __host__ MoE() {
    tokens = batch_size * sequence_len;
    num_experts = num_local_experts * nvshmem.npes;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * tokens * k));
    d_x = static_cast<float*>(nvshmem_malloc(sizeof(float) * tokens * input_dim));
    d_y = static_cast<float*>(nvshmem_malloc(sizeof(float) * tokens * output_dim));
  }

  __host__ ~MoE() {
    nvshmem_free(d_x);
    nvshmem_free(d_y);
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Run() {
    cudaLaunchConfig_t cfg{0};
    int block_dim = std::min(tokens, prop.maxThreadsPerBlock);
    int grid_dim = (tokens + block_dim - 1) / block_dim;
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, Forward, d_x, d_y, d_indices, seed, tokens, input_dim, output_dim, num_experts, num_local_experts);
  }
};

int main(int argc, char* argv[]) {
  auto moe = MoE();
  moe.Run();
}
