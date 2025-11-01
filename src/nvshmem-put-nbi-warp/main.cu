#include <cuda.h>
#include <errno.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string.h>
#include <unistd.h>

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

#define KERNEL_CHECK(msg)                                                                                   \
  do {                                                                                                      \
    cudaError_t err = cudaGetLastError();                                                                   \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d] %s got CUDA error: %s\n", __FILE__, __LINE__, msg, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

#define MPI_CHECK(exp)                                                                \
  do {                                                                                \
    int rc = (exp);                                                                   \
    if (MPI_SUCCESS != rc) {                                                          \
      fprintf(stderr, "[%s:%d] MPI failed with error %d \n", __FILE__, __LINE__, rc); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

struct NVSHMEM {
  int world_size;
  int world_rank;
  int local_size;
  int local_rank;
  int mype;
  int npes;
  int mype_node;

  NVSHMEM() = delete;
  NVSHMEM(int argc, char* argv[]) {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    CUDA_CHECK(cudaSetDevice(local_rank));

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    nvshmem_barrier_all();
  }

  ~NVSHMEM() {
    nvshmem_finalize();
    MPI_CHECK(MPI_Finalize());
  }

  friend std::ostream& operator<<(std::ostream& os, NVSHMEM& nvshmem) {
    os << "world_size: " << nvshmem.world_size << " world_rank: " << nvshmem.world_rank << " mype: " << nvshmem.mype
       << " npes: " << nvshmem.npes << " mype_node: " << nvshmem.mype_node;
    return os;
  }
};

__global__ void ring(int* dst, int* src, const int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  // Note that warpSize is predefined variable in CUDA kernel
  src[idx] = mype;
  for (int widx = 0; widx < n; widx += warpSize) {
    int size = min(warpSize, n - widx);
    nvshmemx_int_put_nbi_warp(dst + widx, src + widx, size, peer);
  }
}

struct NVSHMEMRun {
  static constexpr int BUFSIZE = 64;
  NVSHMEM nvshmem;
  cudaStream_t stream;
  int* dst;
  int* src;

  NVSHMEMRun() = delete;

  __host__ NVSHMEMRun(int argc, char* argv[]) : nvshmem{argc, argv} {
    std::cout << nvshmem << std::endl;
    CUDA_CHECK(cudaStreamCreate(&stream));
    dst = static_cast<int*>(nvshmem_malloc(sizeof(int) * BUFSIZE));
    src = static_cast<int*>(nvshmem_malloc(sizeof(int) * BUFSIZE));
  }

  __host__ ~NVSHMEMRun() {
    nvshmem_free(src);
    nvshmem_free(dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Launch() {
    constexpr int BLOCK_SIZE = 32;
    auto grid_dim = (BUFSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto block_dim = BLOCK_SIZE;
    ring<<<grid_dim, block_dim, 0, stream>>>(dst, src, BUFSIZE);
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy data from GPU to CPU
    int msg[BUFSIZE] = {0};
    CUDA_CHECK(cudaMemcpyAsync(&msg, dst, sizeof(int) * BUFSIZE, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << nvshmem.mype << " recv: ";
    for (size_t i = 0; i < BUFSIZE; ++i) std::cout << msg[i];
    std::cout << std::endl;
  }
};

int main(int argc, char* argv[]) {
  auto runner = NVSHMEMRun(argc, argv);
  runner.Launch();
}
