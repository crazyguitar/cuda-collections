#include <errno.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

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
  int mype;
  int npes;
  int mype_node;

  NVSHMEM() {
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));
  }

  ~NVSHMEM() { nvshmem_finalize(); }

  friend std::ostream& operator<<(std::ostream& os, NVSHMEM& nvshmem) {
    auto mype = nvshmem.mype;
    auto npes = nvshmem.npes;
    auto mype_node = nvshmem.mype_node;
    return os << " mype: " << mype << " npes: " << npes << " mype_node: " << mype_node;
  }
};

struct Rand {
  const unsigned long seed = 0;
  std::mt19937 generator;

  Rand() : generator{seed} {};
  __host__ Rand(unsigned long seed) : seed{seed}, generator{seed} {}

  __host__ std::vector<int> RandInts(int mi, int mx, int k) {
    auto numbers = std::vector<int>(mx - mi);
    std::iota(numbers.begin(), numbers.end(), mi);
    std::shuffle(numbers.begin(), numbers.end(), generator);
    return std::vector<int>(numbers.begin(), numbers.begin() + k);
  }

  __host__ void RandIndices(int* indices, int m, int k, int mi, int mx) {
    for (int i = 0; i < m; ++i) {
      auto rand = RandInts(mi, mx, k);
      for (int j = 0; j < k; ++j) {
        indices[i * k + j] = rand[j];
      }
    }
  }
};

__global__ void MoECount(int* tokens_per_expert, int* recv_per_pe, int* send_per_pe, int mype, int npes, int num_experts, int num_local_experts) {
  int* tokens_count = &tokens_per_expert[mype * num_experts];
  for (int pe = threadIdx.x; pe < npes; pe += blockDim.x) {
    int sum = 0;
    for (int i = 0; i < num_local_experts; ++i) {
      sum += tokens_count[pe * num_local_experts + i];
    }
    send_per_pe[pe] = sum;
  }
  __syncthreads();
  // exchange local tokens_per_expert
  for (int pe = threadIdx.x; pe < npes; pe += blockDim.x) {
    auto src = &tokens_per_expert[mype * num_experts];
    auto dst = src;
    nvshmem_int_put(dst, src, num_experts, pe);
  }

  nvshmem_barrier_all();
  // ensure all nvshmem ops across all PEs are complete; otherwise, recv_per_pe may get 0.
  __syncthreads();
  for (int pe = threadIdx.x; pe < npes; pe += blockDim.x) {
    int sum = 0;
    for (int i = 0; i < num_local_experts; ++i) {
      int expert = mype * num_local_experts + i;
      sum += tokens_per_expert[pe * num_experts + expert];
    }
    recv_per_pe[pe] = sum;
  }
}

struct MoE {
  constexpr static int batch_size = 2;
  constexpr static int sequence_len = 32;
  constexpr static int num_local_experts = 2;
  constexpr static int k = 2;
  constexpr static unsigned long seed = 123;

  NVSHMEM nvshmem;
  Rand rand;
  cudaDeviceProp prop;
  cudaStream_t stream;

  int tokens;
  int num_experts;
  int max_tokens;
  int* indices;
  int* tokens_per_expert;
  int* recv_per_pe;
  int* send_per_pe;
  int* d_tokens_per_expert;
  int* d_recv_per_pe;
  int* d_send_per_pe;

  __host__ MoE() : rand(seed + nvshmem.mype) {
    auto npes = nvshmem.npes;
    auto mype = nvshmem.mype;
    tokens = batch_size * sequence_len;
    max_tokens = npes * tokens;
    num_experts = npes * num_local_experts;
    num_experts = num_local_experts * npes;
    indices = static_cast<int*>(calloc(tokens * k, sizeof(int)));
    tokens_per_expert = static_cast<int*>(calloc(num_experts * npes, sizeof(int)));
    recv_per_pe = static_cast<int*>(calloc(npes, sizeof(int)));
    send_per_pe = static_cast<int*>(calloc(npes, sizeof(int)));
    rand.RandIndices(indices, tokens, k, 0, num_experts);
    Count(indices, tokens_per_expert, tokens, k, num_experts, mype);

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_recv_per_pe, sizeof(int) * npes));
    CUDA_CHECK(cudaMalloc(&d_send_per_pe, sizeof(int) * npes));
    d_tokens_per_expert = static_cast<int*>(nvshmem_malloc(sizeof(int) * npes * num_experts));
    CUDA_CHECK(cudaMemcpy(d_tokens_per_expert, tokens_per_expert, sizeof(int) * npes * num_experts, cudaMemcpyHostToDevice));
  }

  __host__ ~MoE() {
    free(indices);
    free(tokens_per_expert);
    free(recv_per_pe);
    free(send_per_pe);
    nvshmem_free(d_tokens_per_expert);
    CUDA_CHECK(cudaFree(d_recv_per_pe));
    CUDA_CHECK(cudaFree(d_send_per_pe));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Count(int* indices, int* tokens_per_expert, int tokens, int k, int num_experts, int mype) {
    for (int i = 0; i < tokens; ++i) {
      for (int j = 0; j < k; ++j) {
        int expert = indices[i * k + j];
        ++tokens_per_expert[mype * num_experts + expert];
      }
    }
  }

  __host__ void Print() {
    auto mype = nvshmem.mype;
    auto npes = nvshmem.npes;
    if (mype != 0) return;
    std::cout << "tokens_per_expert:\n";
    for (int i = 0; i < npes; ++i) {
      for (int j = 0; j < num_experts; ++j) {
        std::cout << tokens_per_expert[i * num_experts + j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "recv_per_pe:\n";
    for (int i = 0; i < npes; ++i) std::cout << recv_per_pe[i] << " ";
    std::cout << std::endl;

    std::cout << "send_per_pe:\n";
    for (int i = 0; i < npes; ++i) std::cout << send_per_pe[i] << " ";
    std::cout << std::endl;
  }

  __host__ void Run() {
    auto mype = nvshmem.mype;
    auto npes = nvshmem.npes;
    int block_dim = std::min(tokens, prop.maxThreadsPerBlock);
    int grid_dim = 1;
    cudaLaunchConfig_t cfg{0};
    cfg.gridDim = dim3(grid_dim, 1, 1);
    cfg.blockDim = dim3(block_dim, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, MoECount, d_tokens_per_expert, d_recv_per_pe, d_send_per_pe, mype, npes, num_experts, num_local_experts);
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(tokens_per_expert, d_tokens_per_expert, sizeof(int) * npes * num_experts, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(recv_per_pe, d_recv_per_pe, sizeof(int) * npes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(send_per_pe, d_send_per_pe, sizeof(int) * npes, cudaMemcpyDeviceToHost));
    Print();
  }
};

int main(int argc, char* argv[]) {
  auto moe = MoE();
  moe.Run();
}
