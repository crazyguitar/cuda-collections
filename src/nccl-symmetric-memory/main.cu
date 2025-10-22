#include <cuda.h>
#include <mpi.h>
#include <nccl.h>
#include <string.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
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

#define NCCL_CHECK(exp)                                                                                    \
  do {                                                                                                     \
    ncclResult_t rc = (exp);                                                                               \
    if (ncclSuccess != rc) {                                                                               \
      fprintf(stderr, "[%s:%d]" #exp " got NCCL error: %s\n", __FILE__, __LINE__, ncclGetErrorString(rc)); \
      exit(1);                                                                                             \
    }                                                                                                      \
  } while (0)

#define ASSERT(exp)                                                           \
  do {                                                                        \
    if (!(exp)) {                                                             \
      fprintf(stderr, "[%s:%d]" #exp "assertion fail\n", __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

struct NCCL {
  int world_size;
  int world_rank;
  int local_size;
  int local_rank;
  ncclComm_t comm;

  NCCL() = delete;
  NCCL(int argc, char* argv[]) {
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    CUDA_CHECK(cudaSetDevice(local_rank));

    // NCCL initialization
    ncclUniqueId id;
    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));
  }

  ~NCCL() {
    NCCL_CHECK(ncclCommFinalize(comm));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
  }
};

__global__ void init(float* buf, const float x, const int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  buf[idx] = x;
}

struct NCCLRun {
  static constexpr int BUFSIZE = 1024 * 1024 * 1024;  // 1 GB
  NCCL nccl;
  cudaStream_t stream;
  float* sendbuf = nullptr;
  float* recvbuf = nullptr;
  float* buf = nullptr;
  ncclWindow_t sendwin = nullptr;
  ncclWindow_t recvwin = nullptr;

  NCCLRun() = delete;
  __host__ NCCLRun(int argc, char* argv[]) : nccl{argc, argv} {
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Alloc NCCL buffer & Register symmetric memory window
    auto comm = nccl.comm;
    NCCL_CHECK(ncclMemAlloc((void**)&sendbuf, sizeof(float) * BUFSIZE));
    NCCL_CHECK(ncclMemAlloc((void**)&recvbuf, sizeof(float) * BUFSIZE));
    NCCL_CHECK(ncclCommWindowRegister(comm, sendbuf, sizeof(float) * BUFSIZE, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, recvbuf, sizeof(float) * BUFSIZE, &recvwin, NCCL_WIN_COLL_SYMMETRIC));
    buf = (float*)malloc(BUFSIZE * sizeof(float));
  }

  __host__ ~NCCLRun() {
    free(buf);
    NCCL_CHECK(ncclCommWindowDeregister(nccl.comm, sendwin));
    NCCL_CHECK(ncclCommWindowDeregister(nccl.comm, recvwin));
    NCCL_CHECK(ncclMemFree(sendbuf));
    NCCL_CHECK(ncclMemFree(recvbuf));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  __host__ void Launch() {
    constexpr float epslion = 1.0e-8;
    constexpr int BLOCK_SIZE = 32;
    auto comm = nccl.comm;
    auto grid_dim = (BUFSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto block_dim = BLOCK_SIZE;
    init<<<grid_dim, block_dim, 0, stream>>>(sendbuf, nccl.world_rank, BUFSIZE);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    KERNEL_CHECK("Init sendbuf");
    NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, BUFSIZE, ncclFloat, ncclSum, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float expected = (nccl.world_size - 1) * (nccl.world_size) / 2.0;  // (a0 + an) * n / 2
    cudaMemcpy(buf, recvbuf, BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < BUFSIZE; ++i) ASSERT(std::fabs(buf[i] - expected) < epslion);
  }
};

int main(int argc, char* argv[]) {
  auto ncclrun = NCCLRun(argc, argv);
  ncclrun.Launch();
}
