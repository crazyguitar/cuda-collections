#include <cuda.h>
#include <errno.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(exp)                                                                                     \
  do {                                                                                                      \
    cudaError_t err = (exp);                                                                                \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d]" #exp " got CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

struct NVSHMEM {
  int mype;
  int npes;
  int mype_node;

  NVSHMEM() = delete;
  NVSHMEM(int argc, char* argv[]) {
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));
  }

  ~NVSHMEM() { nvshmem_finalize(); }

  friend std::ostream& operator<<(std::ostream& os, NVSHMEM& nvshmem) {
    return os << " mype: " << nvshmem.mype << " npes: " << nvshmem.npes << " mype_node: " << nvshmem.mype_node;
  }
};

__global__ void all2all(int* dst, int* src, int mype, int npes, int dim) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) src[mype * dim + idx] = mype + idx;
  __syncthreads();
  for (int i = threadIdx.x; i < npes; i += blockDim.x) {
    nvshmem_int_put(&dst[mype * dim], &src[mype * dim], dim, i);
  }
}

struct NVSHMEMRun {
  NVSHMEM nvshmem;
  cudaStream_t stream;
  constexpr static int dim = 8;
  int* src;
  int* dst;
  int* res;

  NVSHMEMRun() = delete;

  __host__ NVSHMEMRun(int argc, char* argv[]) : nvshmem{argc, argv} {
    std::cout << nvshmem << std::endl;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto npes = nvshmem.npes;
    src = static_cast<int*>(nvshmem_malloc(sizeof(int) * npes * dim));
    dst = static_cast<int*>(nvshmem_malloc(sizeof(int) * npes * dim));
    res = static_cast<int*>(malloc(sizeof(int) * npes * dim));
  }

  __host__ ~NVSHMEMRun() {
    free(res);
    nvshmem_free(src);
    nvshmem_free(dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Print() {
    auto npes = nvshmem.npes;
    auto mype = nvshmem.mype;
    if (mype != 0) return;
    for (int i = 0; i < npes; ++i) {
      for (int j = 0; j < dim; ++j) std::cout << res[i * dim + j] << " ";
      std::cout << std::endl;
    }
  }

  __host__ void Launch() {
    auto block_dim = std::max(nvshmem.npes, 32);
    auto npes = nvshmem.npes;
    auto mype = nvshmem.mype;
    all2all<<<1, block_dim, 0, stream>>>(dst, src, mype, npes, dim);
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(res, dst, sizeof(int) * npes * dim, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Print();
  }
};

int main(int argc, char* argv[]) {
  auto runner = NVSHMEMRun(argc, argv);
  runner.Launch();
}
