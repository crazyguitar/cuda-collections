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
    return os << "[" << nvshmem.hostname << "] mype: " << nvshmem.mype << " npes: " << nvshmem.npes << " mype_node: " << nvshmem.mype_node;
  }
};

__global__ void Init(float* p, int base, int hidden_dim) {
  int row = blockIdx.x;
  assert(hidden_dim >= blockDim.x);
  for (int col = threadIdx.x; col < hidden_dim; col += hidden_dim / blockDim.x) {
    p[row * hidden_dim + col] = row + base * 128 + (float)col / hidden_dim;
  }
}

struct MoE {
  constexpr static int k = 2;
  constexpr static int experts = 16;
  constexpr static int hidden_dim = 4096;
  constexpr static int batch_size = 2;
  constexpr static int sequence_len = 192;
  constexpr static int total_tokens = batch_size * sequence_len;
  constexpr static int total_rows = total_tokens * k;
  constexpr static int num_elems = total_rows * hidden_dim;
  constexpr static int capacity = 2;

  int* d_expanded_src_row;
  int* d_expert_for_expanded_src_row;
  int* d_expert_pos;
  int* d_expert_count;
  int* d_expert_offset;
  float* d_sendbuf;
  float* d_recvbuf;
  NVSHMEM nvshmem;

  __host__ MoE() {
    auto npes = nvshmem.npes;
    CUDA_CHECK(cudaMalloc(&d_expert_pos, sizeof(int) * npes * experts * total_rows));
    CUDA_CHECK(cudaMalloc(&d_expanded_src_row, sizeof(int) * total_rows));
    CUDA_CHECK(cudaMalloc(&d_expert_for_expanded_src_row, sizeof(int) * total_rows));
    d_expert_count = static_cast<int*>(nvshmem_malloc(sizeof(int) * experts * total_rows));
    d_expert_offset = static_cast<int*>(nvshmem_malloc(sizeof(int) * experts));
    d_sendbuf = static_cast<float*>(nvshmem_malloc(sizeof(float) * total_tokens * hidden_dim));
    d_recvbuf = static_cast<float*>(nvshmem_malloc(sizeof(float) * total_rows * hidden_dim * capacity));
    Init<<<total_tokens, hidden_dim>>>(d_sendbuf, nvshmem.mype, hidden_dim);
    Permute();
  }

  __host__ __forceinline__ void Permute() {
    std::vector<std::pair<int, int>> expert_map(total_rows);
    std::vector<int> expert_count(experts);
    std::vector<int> expert_offset(experts);
    std::vector<int> expanded_src_row(total_rows);
    std::vector<int> expert_for_expanded_src_row(total_rows);
    int cur_expert = 0;

    for (int i = 0; i < expert_map.size() / 2; ++i) {
      for (int j = 0; j < k; ++j) {
        auto selected = cur_expert % experts;
        ++cur_expert;
        expert_map[i + j * total_tokens] = {selected, i + j * total_tokens};
        ++expert_count[selected];
      }
    }

    int prev = 0;
    for (int i = 0; i < experts; ++i) {
      prev += expert_count[i];
      expert_offset[i] = prev;
    }

    std::sort(expert_map.begin(), expert_map.end());
    for (int i = 0; i < expert_map.size(); ++i) {
      expanded_src_row[i] = expert_map[i].first;
      expert_for_expanded_src_row[i] = expert_map[i].second;
    }

    auto p = d_expert_count;
    for (int i = 0; i < total_rows; ++i) {
      CUDA_CHECK(cudaMemcpy(p, expert_count.data(), experts * sizeof(int), cudaMemcpyHostToDevice));
      p += experts;
    }

    CUDA_CHECK(cudaMemcpy(d_expert_offset, expert_offset.data(), experts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expanded_src_row, expanded_src_row.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_for_expanded_src_row, expert_for_expanded_src_row.data(), total_rows * sizeof(int), cudaMemcpyHostToDevice));
  }

  __host__ ~MoE() {
    CUDA_CHECK(cudaFree(d_expanded_src_row));
    CUDA_CHECK(cudaFree(d_expert_for_expanded_src_row));
    CUDA_CHECK(cudaFree(d_expert_pos));
    nvshmem_free(d_expert_count);
    nvshmem_free(d_expert_offset);
    nvshmem_free(d_sendbuf);
    nvshmem_free(d_recvbuf);
  }

  friend std::ostream& operator<<(std::ostream& os, const MoE& moe) { return os << moe.nvshmem; }
};

int main(int argc, char* argv[]) {
  auto moe = MoE();
  std::cout << moe << std::endl;
}
