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

__device__ __forceinline__ void InitIndices(curandState& state, int* indices, int k, int tokens, int num_experts) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
    indices[i * k] = curand(&state) % num_experts;
#pragma unroll
    for (int j = 1; j < k; ++j) {
      indices[j + i * k] = (indices[i * k] + 1) % num_experts;
    }
  }
}

__device__ __forceinline__ void InitTokens(curandState& state, float* x, int tokens, int input_dim) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < input_dim; ++j) {
      x[j + i * input_dim] = curand_uniform(&state);
    }
  }
}

__device__ __forceinline__ void Count(
    int* indices,
    int* tokens_per_expert,
    int* tokens_per_pe,
    int tokens,
    int k,
    int num_experts,
    int num_local_experts,
    int mype,
    int npes
) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < k; ++j) {
      int expert = indices[j + i * k];
      atomicAdd(&tokens_per_expert[expert + num_experts * mype], 1);
    }
  }
  __syncthreads();
  for (int peer = threadIdx.x; peer < npes; peer += blockDim.x) {
    if (peer == mype) continue;
    int* dst = &tokens_per_expert[num_experts * peer];
    int* src = &tokens_per_expert[num_experts * mype];
    nvshmem_int_put(dst, src, num_experts, peer);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    int pe = i / num_local_experts;
    int offset = i % num_local_experts;
    int expert = pe + offset;
    for (int j = 0; j < npes; ++j) {
      atomicAdd(&tokens_per_pe[pe], tokens_per_expert[expert + j * num_experts]);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void Dispatch(
    float* input_tokens,
    float* send_tokens,
    float* recv_tokens,
    int* indices,
    int* tokens_per_expert,
    int mype,
    int npes
) {
  float* local_tokens_per_expert = &tokens_per_expert[mype * num_experts];


}

__global__ void MoEKernel(
    float* input_tokens,
    float* send_tokens,
    float* recv_tokens,
    int* indices,
    int* tokens_per_expert,
    int* tokens_per_pe,
    int seed,
    int k,
    int tokens,
    int max_tokens,
    int input_dim,
    int num_experts,
    int num_local_experts,
    int mype,
    int npes
) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(seed, idx, 0, &rand_state);
  InitIndices(rand_state, indices, k, tokens, num_experts);
  InitTokens(rand_state, input_tokens, tokens, input_dim);
  __syncthreads();

  Count(indices, tokens_per_expert, tokens_per_pe, tokens, k, num_experts, num_local_experts, mype, npes);
  Dispatch(input_tokens, send_tokens, recv_tokens, indices, tokens_per_expert, mype, npes);
}

struct MoE {
  constexpr static int batch_size = 4;
  constexpr static int sequence_len = 1024;
  constexpr static int input_dim = 512;
  constexpr static int num_local_experts = 2;
  constexpr static int k = 2;
  constexpr static int seed = 123;
  NVSHMEM nvshmem;
  cudaDeviceProp prop;
  cudaStream_t stream;

  /* MoE attr */
  int tokens;
  int num_experts;
  int max_tokens;
  int* d_indices;
  int* d_tokens_per_expert;
  int* d_tokens_per_pe;
  float* d_input_tokens;
  float* d_send_tokens;
  float* d_recv_tokens;

  __host__ MoE() {
    auto npes = nvshmem.npes;
    tokens = batch_size * sequence_len;
    max_tokens = npes * tokens;
    num_experts = num_local_experts * nvshmem.npes;

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * tokens * k));
    CUDA_CHECK(cudaMalloc(&d_tokens_per_pe, sizeof(int) * npes));
    CUDA_CHECK(cudaMalloc(&d_input_tokens, sizeof(int) * tokens * input_dim));
    d_send_tokens = static_cast<float*>(nvshmem_malloc(sizeof(float) * k * tokens * input_dim));
    d_recv_tokens = static_cast<float*>(nvshmem_malloc(sizeof(float) * npes * max_tokens * input_dim));
    d_tokens_per_expert = static_cast<int*>(nvshmem_malloc(sizeof(int) * npes * num_experts));
  }

  __host__ ~MoE() {
    nvshmem_free(d_send_tokens);
    nvshmem_free(d_recv_tokens);
    nvshmem_free(d_tokens_per_expert);
    CUDA_CHECK(cudaFree(d_tokens_per_pe));
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
        &cfg,
        MoEKernel,
        d_input_tokens,
        d_send_tokens,
        d_recv_tokens,
        d_indices,
        d_tokens_per_expert,
        d_tokens_per_pe,
        seed,
        k,
        tokens,
        max_tokens,
        input_dim,
        num_experts,
        num_local_experts,
        mype,
        npes
    );
  }
};

int main(int argc, char* argv[]) {
  auto moe = MoE();
  moe.Run();
}
