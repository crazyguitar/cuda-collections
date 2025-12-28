/**
 * CUDA IPC Ring Communication (Multi-Process)
 *
 * Comparison: IPC vs P2P Direct
 * ┌─────────────────────┬──────────────────────────────┬──────────────────────────────┐
 * │                     │ IPC (this file)              │ P2P Direct (p2p-nvlink)      │
 * ├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
 * │ Process model       │ Multi-process (MPI)          │ Single process, multi-GPU    │
 * │ API                 │ cudaIpcGetMemHandle          │ cudaDeviceEnablePeerAccess   │
 * │ Setup               │ Exchange handles via MPI     │ Just enable peer access      │
 * │ Launch              │ mpirun -np N ./p2p-ipc       │ ./p2p-nvlink                 │
 * │ Fault isolation     │ Process isolation            │ None                         │
 * │ Use case            │ HPC/ML distributed training  │ Simple multi-GPU apps        │
 * │ NCCL/NVSHMEM        │ Uses IPC internally          │ Not compatible               │
 * ├─────────────────────┴──────────────────────────────┴──────────────────────────────┤
 * │ Both use same hardware path: NVLink (if available) or PCIe                        │
 * └───────────────────────────────────────────────────────────────────────────────────┘
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi/mpi.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(exp)                                                                                     \
  do {                                                                                                      \
    cudaError_t err = (exp);                                                                                \
    if (err != cudaSuccess) {                                                                               \
      fprintf(stderr, "[%s:%d]" #exp " got CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

__global__ void write_to_peer(int* __restrict__ peer_buf, int rank, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) peer_buf[idx] = rank * 1000 + idx;
}

__global__ void verify_buffer(int* __restrict__ buf, int expected_rank, size_t len, int* errors) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    int expected = expected_rank * 1000 + idx;
    if (buf[idx] != expected) atomicAdd(errors, 1);
  }
}

struct Peer {
  static constexpr size_t kBufsize = 1024;

  int npes;
  int pe;
  int device;
  int* d_buf;
  cudaIpcMemHandle_t me;
  std::vector<cudaIpcMemHandle_t> handles;
  std::vector<int*> peers;

  Peer() {
    auto& mpi = MPI::Get();
    device = mpi.GetLocalRank();
    pe = mpi.GetLocalRank();
    npes = mpi.GetLocalSize();

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int) * kBufsize));
    CUDA_CHECK(cudaMemset(d_buf, 0, kBufsize * sizeof(int)));
    CUDA_CHECK(cudaIpcGetMemHandle(&me, d_buf));

    handles.resize(npes);
    MPI_Allgather(&me, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

    peers.resize(npes);
    for (int i = 0; i < npes; ++i) {
      if (i == pe) {
        peers[i] = d_buf;
      } else {
        CUDA_CHECK(cudaIpcOpenMemHandle((void**)&peers[i], handles[i], cudaIpcMemLazyEnablePeerAccess));
      }
    }
  }

  ~Peer() {
    for (int i = 0; i < npes; ++i) {
      if (i != pe) CUDA_CHECK(cudaIpcCloseMemHandle(peers[i]));
    }
    CUDA_CHECK(cudaFree(d_buf));
  }
};

struct Test {
  int* d_err;

  Test() {
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
  }

  ~Test() { CUDA_CHECK(cudaFree(d_err)); }

  void Run() {
    auto peer = Peer();
    int block = 256;
    int grid = (Peer::kBufsize + block - 1) / block;
    int target = (peer.pe + 1) % peer.npes;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block);

    LAUNCH_KERNEL(&cfg, write_to_peer, peer.peers[target], peer.pe, Peer::kBufsize);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    int source = (peer.pe - 1 + peer.npes) % peer.npes;
    LAUNCH_KERNEL(&cfg, verify_buffer, peer.d_buf, source, Peer::kBufsize, d_err);
    CUDA_CHECK(cudaDeviceSynchronize());

    int err = 0;
    CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[Rank %d] Written by rank %d, errors: %d %s\n", peer.pe, source, err, err == 0 ? "OK" : "FAIL");

    if (peer.pe == 0) {
      int buf[16];
      CUDA_CHECK(cudaMemcpy(buf, peer.d_buf, sizeof(buf), cudaMemcpyDeviceToHost));
      printf("[Rank 0] First 16 elements:");
      for (int i = 0; i < 16; ++i) printf(" %d", buf[i]);
      printf("\n");
    }
  }
};

/**
 * Rank 0                                    Rank 1                                   Rank N-1
 *   |                                         |                                         |
 *   |-- cudaMalloc(d_buf) --------------------|-- cudaMalloc(d_buf) --------------------|
 *   |                                         |                                         |
 *   |-- cudaIpcGetMemHandle(handle) ----------|-- cudaIpcGetMemHandle(handle) ----------|
 *   |                                         |                                         |
 *   |<------------------ MPI_Allgather (exchange all handles) ------------------------->|
 *   |                                         |                                         |
 *   |-- cudaIpcOpenMemHandle(peers[]) --------|-- cudaIpcOpenMemHandle(peers[]) --------|
 *   |                                         |                                         |
 *   |-- write_to_peer(peers[1]) ------------->|    (GPU 0 writes to GPU 1's buf)        |
 *   |                                         |-- write_to_peer(peers[2]) ------------->|
 *   |    (GPU N-1 writes to GPU 0's buf) <----|-----------------------------------------|
 *   |                                         |                                         |
 *   |<------------------ MPI_Barrier (sync all) --------------------------------------->|
 *   |                                         |                                         |
 *   |-- verify_buffer(d_buf) -----------------|-- verify_buffer(d_buf) -----------------|
 *   |    (check data from rank N-1)           |    (check data from rank 0)             |
 *   |                                         |                                         |
 *   |-- cudaIpcCloseMemHandle ----------------|-- cudaIpcCloseMemHandle ----------------|
 */
// mpirun -np 8 ./build/src/ipc/ipc
int main(int argc, char* argv[]) {
  Test test;
  test.Run();
}
