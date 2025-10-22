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

#define MPI_CHECK(exp)                                                                \
  do {                                                                                \
    int rc = (exp);                                                                   \
    if (MPI_SUCCESS != rc) {                                                          \
      fprintf(stderr, "[%s:%d] MPI failed with error %d \n", __FILE__, __LINE__, rc); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

#define NVSHMEM_CHECK(exp)                                                                \
  do {                                                                                    \
    int rc = (exp);                                                                       \
    if (NVSHMEMX_SUCCESS != rc) {                                                         \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, rc); \
      exit(1);                                                                            \
    }                                                                                     \
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
    os << "world_size: " << nvshmem.world_size << " world_rank: " << nvshmem.world_rank << " mype: " << nvshmem.mype << " npes: " << nvshmem.npes
       << " mype_node: " << nvshmem.mype_node;
    return os;
  }
};

__global__ void ring(int* dst, int* src) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  *src = mype;
  // Note that both dst and src should be on symmetric memory; otherwise, the
  // process will receive segfault.
  nvshmem_int_put_nbi(dst, src, 1, peer);
  nvshmem_fence();
  nvshmem_quiet();
}

struct NVSHMEMRun {
  NVSHMEM nvshmem;
  cudaStream_t stream;
  int* dst;
  int* src;

  NVSHMEMRun() = delete;

  __host__ NVSHMEMRun(int argc, char* argv[]) : nvshmem{argc, argv} {
    std::cout << nvshmem << std::endl;
    CUDA_CHECK(cudaStreamCreate(&stream));
    dst = static_cast<int*>(nvshmem_malloc(sizeof(int)));
    src = static_cast<int*>(nvshmem_malloc(sizeof(int)));
  }

  __host__ ~NVSHMEMRun() {
    nvshmem_free(src);
    nvshmem_free(dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Launch() {
    void* args[] = {&dst, &src};
    // Base on the document, when the CUDA kernel utilize NVSHMEM synchronization,
    // nvshmemx_collective_launch must be used.
    // ref: https://docs.nvidia.com/nvshmem/archives/nvshmem-203/api/docs/api/launch.html?highlight=nvshmemx_collective_launch
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void*)ring, 1, 1, args, 0, stream));
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy data from GPU to CPU
    int msg;
    CUDA_CHECK(cudaMemcpyAsync(&msg, dst, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << nvshmem.mype << " recv: " << msg << std::endl;
  }
};

int main(int argc, char* argv[]) {
  auto runner = NVSHMEMRun(argc, argv);
  runner.Launch();
}
