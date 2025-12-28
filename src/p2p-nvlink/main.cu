/**
 * P2P Direct Access Ring Communication (Single-Process)
 *
 * How P2P Direct Works:
 * ─────────────────────
 * 1. Single process owns all GPUs in the system
 * 2. cudaDeviceEnablePeerAccess() creates a mapping in GPU's page table
 *    allowing direct load/store to peer GPU memory
 * 3. GPU kernel can directly dereference peer's pointer (no memcpy needed)
 *
 * Why MPI Cannot Be Used With P2P Direct:
 * ───────────────────────────────────────
 * cudaDeviceEnablePeerAccess() only works within the SAME process.
 *
 *   Single Process (P2P Direct works):
 *   ┌─────────────────────────────────────────┐
 *   │ Process                                 │
 *   │  ┌─────────┐         ┌─────────┐        │
 *   │  │  GPU 0  │ ══════▶ │  GPU 1  │        │  ✓ Same address space
 *   │  │  ptr=A  │         │  ptr=B  │        │  ✓ Can dereference B from GPU 0
 *   │  └─────────┘         └─────────┘        │
 *   └─────────────────────────────────────────┘
 *
 *   MPI Multi-Process (P2P Direct FAILS):
 *   ┌───────────────────┐   ┌───────────────────┐
 *   │ Process 0 (Rank 0)│   │ Process 1 (Rank 1)│
 *   │  ┌─────────┐      │   │      ┌─────────┐  │
 *   │  │  GPU 0  │ ─ ─ ─│─ ─│─ ─ ▶ │  GPU 1  │  │  ✗ Different address spaces
 *   │  │  ptr=A  │      │   │      │  ptr=B  │  │  ✗ ptr B is invalid in Process 0
 *   │  └─────────┘      │   │      └─────────┘  │
 *   └───────────────────┘   └───────────────────┘
 *
 * Each MPI rank is a separate OS process with isolated virtual address space.
 * A pointer allocated in Process 1 is meaningless in Process 0.
 * → Use IPC (cudaIpcGetMemHandle) to share memory across processes.
 *
 * Memory Access Path:
 * ───────────────────
 *   GPU 0 Thread                         GPU 1 HBM
 *       │                                    │
 *       │  st.global [peer_addr], val        │
 *       ▼                                    │
 *   ┌─────────┐                              │
 *   │ L2 Cache│ (miss - not local)           │
 *   └────┬────┘                              │
 *        │                                   │
 *        ▼                                   │
 *   ┌─────────┐    NVLink/PCIe          ┌────▼────┐
 *   │ NVLink  │ ═══════════════════════▶│ NVLink  │
 *   │ or PCIe │                         │ or PCIe │
 *   └─────────┘                         └────┬────┘
 *                                            │
 *                                       ┌────▼────┐
 *                                       │ L2 Cache│
 *                                       └────┬────┘
 *                                            │
 *                                       ┌────▼────┐
 *                                       │   HBM   │ ← data written here
 *                                       └─────────┘
 *
 * Comparison: P2P Direct vs IPC
 * ┌─────────────────────┬──────────────────────────────┬──────────────────────────────┐
 * │                     │ P2P Direct (this file)       │ IPC (p2p-ipc)                │
 * ├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
 * │ Process model       │ Single process, multi-GPU    │ Multi-process (MPI)          │
 * │ API                 │ cudaDeviceEnablePeerAccess   │ cudaIpcGetMemHandle          │
 * │ Setup               │ Just enable peer access      │ Exchange handles via MPI     │
 * │ Launch              │ ./p2p-nvlink                 │ mpirun -np N ./p2p-ipc       │
 * │ Fault isolation     │ None                         │ Process isolation            │
 * │ Use case            │ Simple multi-GPU apps        │ HPC/ML distributed training  │
 * │ NCCL/NVSHMEM        │ Not compatible               │ Uses IPC internally          │
 * ├─────────────────────┴──────────────────────────────┴──────────────────────────────┤
 * │ Both use same hardware path: NVLink (if available) or PCIe                        │
 * └───────────────────────────────────────────────────────────────────────────────────┘
 */

#include <cuda.h>
#include <cuda_runtime.h>

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
  int* d_buf;
  std::vector<int*> peers;

  Peer(int gpu_id, int num_gpus) : pe(gpu_id), npes(num_gpus) {
    CUDA_CHECK(cudaSetDevice(pe));
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int) * kBufsize));
    CUDA_CHECK(cudaMemset(d_buf, 0, kBufsize * sizeof(int)));
  }

  ~Peer() {
    CUDA_CHECK(cudaSetDevice(pe));
    for (int i = 0; i < npes; ++i) {
      if (i != pe) cudaDeviceDisablePeerAccess(i);
    }
    CUDA_CHECK(cudaFree(d_buf));
  }

  void EnablePeerAccess() {
    CUDA_CHECK(cudaSetDevice(pe));
    for (int i = 0; i < npes; ++i) {
      if (i != pe) {
        int canAccess;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, pe, i));
        if (canAccess) {
          CUDA_CHECK(cudaDeviceEnablePeerAccess(i, 0));
        } else {
          fprintf(stderr, "Warning: GPU %d cannot access GPU %d\n", pe, i);
        }
      }
    }
  }

  void SetPeers(const std::vector<int*>& all_bufs) { peers = all_bufs; }
};

struct Test {
  int npes;
  std::vector<Peer*> peers;
  std::vector<int*> d_errs;

  Test() {
    CUDA_CHECK(cudaGetDeviceCount(&npes));
    printf("Found %d GPUs\n", npes);

    // Create peers
    for (int i = 0; i < npes; ++i) {
      peers.push_back(new Peer(i, npes));
    }

    // Collect all buffers
    std::vector<int*> all_bufs(npes);
    for (int i = 0; i < npes; ++i) {
      all_bufs[i] = peers[i]->d_buf;
    }

    // Enable P2P and set peer pointers
    for (int i = 0; i < npes; ++i) {
      peers[i]->EnablePeerAccess();
      peers[i]->SetPeers(all_bufs);
    }

    // Allocate error counters
    d_errs.resize(npes);
    for (int i = 0; i < npes; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaMalloc(&d_errs[i], sizeof(int)));
      CUDA_CHECK(cudaMemset(d_errs[i], 0, sizeof(int)));
    }
  }

  ~Test() {
    for (int i = 0; i < npes; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaFree(d_errs[i]));
      delete peers[i];
    }
  }

  void Run() {
    int block = 256;
    int grid = (Peer::kBufsize + block - 1) / block;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block);

    // Ring: each GPU writes to next GPU's buffer
    for (int i = 0; i < npes; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      int target = (i + 1) % npes;
      LAUNCH_KERNEL(&cfg, write_to_peer, peers[i]->peers[target], i, Peer::kBufsize);
    }

    // Sync all
    for (int i = 0; i < npes; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Verify
    for (int i = 0; i < npes; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      int source = (i - 1 + npes) % npes;
      LAUNCH_KERNEL(&cfg, verify_buffer, peers[i]->d_buf, source, Peer::kBufsize, d_errs[i]);
    }

    // Check results
    for (int i = 0; i < npes; ++i) {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaDeviceSynchronize());
      int source = (i - 1 + npes) % npes;
      int err = 0;
      CUDA_CHECK(cudaMemcpy(&err, d_errs[i], sizeof(int), cudaMemcpyDeviceToHost));
      printf("[GPU %d] Written by GPU %d, errors: %d %s\n", i, source, err, err == 0 ? "OK" : "FAIL");
    }

    // Print first 16 elements from GPU 0
    int buf[16];
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(buf, peers[0]->d_buf, sizeof(buf), cudaMemcpyDeviceToHost));
    printf("[GPU 0] First 16 elements:");
    for (int i = 0; i < 16; ++i) printf(" %d", buf[i]);
    printf("\n");
  }
};

/**
 * GPU 0                                     GPU 1                                    GPU N-1
 *   |                                         |                                         |
 *   |-- cudaMalloc(d_buf) --------------------|-- cudaMalloc(d_buf) --------------------|
 *   |                                         |                                         |
 *   |-- cudaDeviceEnablePeerAccess(all) ------|-- cudaDeviceEnablePeerAccess(all) ------|
 *   |                                         |                                         |
 *   |-- write_to_peer(peers[1]) ------------->|    (GPU 0 writes to GPU 1's buf)        |
 *   |                                         |-- write_to_peer(peers[2]) ------------->|
 *   |    (GPU N-1 writes to GPU 0's buf) <----|-----------------------------------------|
 *   |                                         |                                         |
 *   |-- cudaDeviceSynchronize ----------------|-- cudaDeviceSynchronize ----------------|
 *   |                                         |                                         |
 *   |-- verify_buffer(d_buf) -----------------|-- verify_buffer(d_buf) -----------------|
 *   |    (check data from GPU N-1)            |    (check data from GPU 0)              |
 *   |                                         |                                         |
 *   |-- cudaDeviceDisablePeerAccess ----------|-- cudaDeviceDisablePeerAccess ----------|
 */
// ./build/src/p2p-nvlink/p2p-nvlink
int main() {
  Test test;
  test.Run();
}
