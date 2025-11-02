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

/**
 * @brief Initialize routing indices for tokens to experts using random assignment
 * @param state Random state for generation
 * @param indices Output array for expert indices
 * @param k Number of experts per token
 * @param tokens Number of tokens
 * @param num_experts Total number of experts
 */
__device__ __forceinline__ void InitIndices(curandState& state, int* indices, int k, int tokens, int num_experts) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
    indices[i * k] = curand(&state) % num_experts;
#pragma unroll
    for (int j = 1; j < k; ++j) {
      indices[j + i * k] = (indices[i * k] + 1) % num_experts;
    }
  }
}

/**
 * @brief Initialize input tokens with random values
 * @param state Random state for generation
 * @param x Output token array
 * @param tokens Number of tokens
 * @param input_dim Token dimension
 */
__device__ __forceinline__ void InitTokens(curandState& state, float* x, int tokens, int input_dim) {
  for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < input_dim; ++j) {
      x[j + i * input_dim] = curand_uniform(&state);
    }
  }
}

/**
 * @brief Count tokens per expert and distribute counts across all PEs
 * @param indices Expert assignment indices
 * @param tokens_per_expert Output count array
 * @param tokens Number of tokens
 * @param k Number of experts per token
 * @param num_experts Total number of experts
 * @param mype Current PE ID
 * @param npes Total number of PEs
 */
__device__ __forceinline__ void Count(int* indices, int* tokens_per_expert, int tokens, int k, int num_experts, int mype, int npes) {
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    tokens_per_expert[i + num_experts * mype] = 0;
  }
  __syncthreads();

  // Single thread does the counting to avoid race conditions
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < tokens; i++) {
      for (int j = 0; j < k; j++) {
        int expert = indices[j + i * k];
        ++tokens_per_expert[expert + num_experts * mype];
      }
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
}

/**
 * @brief Permute input tokens into send buffer based on expert assignments
 * @param input_tokens Input token array
 * @param send_tokens Output send buffer
 * @param indices Expert assignment indices
 * @param tokens_per_expert Token count per expert
 * @param shared_expert_offset Shared memory for expert offsets
 * @param shared_token_offset Shared memory for token offsets
 * @param k Number of experts per token
 * @param tokens Number of tokens
 * @param input_dim Token dimension
 * @param num_experts Total number of experts
 * @param mype Current PE ID
 */
__device__ __forceinline__ void Permute(
    float* input_tokens,
    float* send_tokens,
    int* indices,
    int* tokens_per_expert,
    int* shared_expert_offset,
    int* shared_token_offset,
    int k,
    int tokens,
    int input_dim,
    int num_experts,
    int mype
) {
  int* count = &tokens_per_expert[mype * num_experts];
  if (threadIdx.x == 0) {
    int prev = 0;
    for (int i = 0; i < num_experts; ++i) {
      shared_expert_offset[i] = prev;
      prev += count[i];
    }

    for (int i = 0; i < tokens; ++i) {
      for (int j = 0; j < k; ++j) {
        auto expert = indices[i * k + j];
        shared_token_offset[i * k + j] = shared_expert_offset[expert];
        ++shared_expert_offset[expert];
      }
    }
  }
  __syncthreads();

  for (int i = 0; i < tokens; ++i) {
    float* token = &input_tokens[i * input_dim];
    for (int j = 0; j < k; ++j) {
      auto offset = shared_token_offset[i * k + j];
      auto row = offset * input_dim;
      for (int x = threadIdx.x; x < input_dim; x += blockDim.x) {
        send_tokens[row + x] = token[x];
      }
    }
  }
  __syncthreads();
}

/**
 * @brief Dispatch tokens to appropriate experts across PEs using NVSHMEM
 * @param input_tokens Input token array
 * @param send_tokens Send buffer
 * @param recv_tokens Receive buffer
 * @param indices Expert assignment indices
 * @param tokens_per_expert Token count per expert
 * @param shared_expert_offset Shared memory for expert offsets
 * @param shared_token_offset Shared memory for token offsets
 * @param k Number of experts per token
 * @param tokens Number of tokens
 * @param input_dim Token dimension
 * @param max_tokens Maximum tokens per expert
 * @param num_experts Total number of experts
 * @param num_local_experts Number of local experts
 * @param mype Current PE ID
 * @param npes Total number of PEs
 */
__device__ __forceinline__ void Dispatch(
    float* input_tokens,
    float* send_tokens,
    float* recv_tokens,
    int* indices,
    int* tokens_per_expert,
    int* shared_expert_offset,
    int* shared_token_offset,
    int k,
    int tokens,
    int input_dim,
    int max_tokens,
    int num_experts,
    int num_local_experts,
    int mype,
    int npes
) {
  Permute(input_tokens, send_tokens, indices, tokens_per_expert, shared_expert_offset, shared_token_offset, k, tokens, input_dim, num_experts, mype);

  int* count = &tokens_per_expert[mype * num_experts];
  if (threadIdx.x == 0) {
    int prev = 0;
    for (int i = 0; i < num_experts; ++i) {
      shared_expert_offset[i] = prev;
      prev += count[i];
    }
  }
  __syncthreads();

  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    auto peer = expert / num_local_experts;
    if (peer == mype) continue;
    auto offset = shared_expert_offset[expert];
    auto elem = count[expert] * input_dim;
    auto src = &send_tokens[offset * input_dim];
    auto dst = &recv_tokens[expert * max_tokens * input_dim];
    nvshmem_float_put(dst, src, elem, peer);
  }

  for (int i = 0; i < num_local_experts; ++i) {
    auto expert = mype * num_local_experts + i;
    auto offset = shared_expert_offset[expert];
    auto elem = count[expert] * input_dim;
    auto src = &send_tokens[offset * input_dim];
    auto dst = &recv_tokens[expert * max_tokens * input_dim];
    for (int i = threadIdx.x; i < elem; i += blockDim.x) {
      dst[i] = src[i];
    }
  }
}

/**
 * @brief Main MoE kernel that handles token routing and expert dispatch
 * @param input_tokens Input token array
 * @param send_tokens Send buffer
 * @param recv_tokens Receive buffer
 * @param indices Expert assignment indices
 * @param tokens_per_expert Token count per expert
 * @param seed Random seed
 * @param k Number of experts per token
 * @param tokens Number of tokens
 * @param input_dim Token dimension
 * @param max_tokens Maximum tokens per expert
 * @param num_experts Total number of experts
 * @param num_local_experts Number of local experts
 * @param mype Current PE ID
 * @param npes Total number of PEs
 */
__global__ void MoEKernel(
    float* input_tokens,
    float* send_tokens,
    float* recv_tokens,
    int* indices,
    int* tokens_per_expert,
    int seed,
    int k,
    int tokens,
    int input_dim,
    int max_tokens,
    int num_experts,
    int num_local_experts,
    int mype,
    int npes
) {
  extern __shared__ char shared_buffer[];
  int* shared_expert_offset = (int*)shared_buffer;
  int* shared_token_offset = (int*)(shared_expert_offset + num_experts);

  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(seed, idx, 0, &rand_state);
  InitIndices(rand_state, indices, k, tokens, num_experts);
  InitTokens(rand_state, input_tokens, tokens, input_dim);
  __syncthreads();

  Count(indices, tokens_per_expert, tokens, k, num_experts, mype, npes);
  Dispatch(
      input_tokens,
      send_tokens,
      recv_tokens,
      indices,
      tokens_per_expert,
      shared_expert_offset,
      shared_token_offset,
      k,
      tokens,
      input_dim,
      max_tokens,
      num_experts,
      num_local_experts,
      mype,
      npes
  );
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
  float* d_input_tokens;
  float* d_send_tokens;
  float* d_recv_tokens;

  /**
   * @brief Initialize MoE system with memory allocation and NVSHMEM setup
   */
  __host__ MoE() {
    auto npes = nvshmem.npes;
    tokens = batch_size * sequence_len;
    max_tokens = npes * tokens;
    num_experts = num_local_experts * nvshmem.npes;

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * tokens * k));
    CUDA_CHECK(cudaMalloc(&d_input_tokens, sizeof(int) * tokens * input_dim));
    d_send_tokens = static_cast<float*>(nvshmem_malloc(sizeof(float) * k * tokens * input_dim));
    d_recv_tokens = static_cast<float*>(nvshmem_malloc(sizeof(float) * num_experts * max_tokens * input_dim));
    d_tokens_per_expert = static_cast<int*>(nvshmem_malloc(sizeof(int) * npes * num_experts));
  }

  /**
   * @brief Clean up allocated memory and resources
   */
  __host__ ~MoE() {
    nvshmem_free(d_send_tokens);
    nvshmem_free(d_recv_tokens);
    nvshmem_free(d_tokens_per_expert);
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  /**
   * @brief Execute MoE kernel with configured parameters
   */
  __host__ void Run() {
    auto shared_expert_offset_size = sizeof(int) * num_experts;
    auto shared_token_offset_size = sizeof(int) * tokens * k;

    auto mype = nvshmem.mype;
    auto npes = nvshmem.npes;
    cudaLaunchConfig_t cfg{0};
    int block_dim = std::min(tokens, prop.maxThreadsPerBlock);
    int grid_dim = 1;  // use single sm
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.dynamicSmemBytes = shared_expert_offset_size + shared_token_offset_size;
    cfg.stream = stream;
    LAUNCH_KERNEL(
        &cfg,
        MoEKernel,
        d_input_tokens,
        d_send_tokens,
        d_recv_tokens,
        d_indices,
        d_tokens_per_expert,
        seed,
        k,
        tokens,
        input_dim,
        max_tokens,
        num_experts,
        num_local_experts,
        mype,
        npes
    );

    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

int main(int argc, char* argv[]) {
  auto moe = MoE();
  moe.Run();
}
