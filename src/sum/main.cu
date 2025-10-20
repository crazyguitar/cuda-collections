#include <cuda.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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

#define ASSERT(exp)                                                           \
  do {                                                                        \
    if (!(exp)) {                                                             \
      fprintf(stderr, "[%s:%d]" #exp "assertion fail\n", __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

__global__ void sum(const auto* A, const auto* B, auto* C, size_t dim) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= dim) return;
  C[idx] = A[idx] + B[idx];
}

void init(auto* M, const size_t dim) {
  for (size_t i = 0; i < dim; ++i) {
    auto x = static_cast<float>(rand()) / RAND_MAX;
    M[i] = static_cast<decltype(*M)>(x);
  }
}

void validate(const auto* A, const auto* B, const auto* C, size_t dim) {
  constexpr float epslion = 1.0e-8;
  for (size_t i = 0; i < dim; ++i) ASSERT(std::fabs(C[i] - (A[i] + B[i])) < epslion);
}

int main(int argc, char* argv[]) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int BUFSIZE = 8192;

  float A_h[BUFSIZE] = {0};
  float B_h[BUFSIZE] = {0};
  float C_h[BUFSIZE] = {0};
  float* A_d = nullptr;
  float* B_d = nullptr;
  float* C_d = nullptr;

  // init memory value
  init(A_h, BUFSIZE);
  init(B_h, BUFSIZE);

  // alloc device memory
  CUDA_CHECK(cudaMalloc(&A_d, sizeof(float) * BUFSIZE));
  CUDA_CHECK(cudaMalloc(&B_d, sizeof(float) * BUFSIZE));
  CUDA_CHECK(cudaMalloc(&C_d, sizeof(float) * BUFSIZE));

  // copy memory from host to device
  CUDA_CHECK(cudaMemcpy(A_d, A_h, BUFSIZE * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, BUFSIZE * sizeof(float), cudaMemcpyHostToDevice));

  int grid_dim = (BUFSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int block_dim = BLOCK_SIZE;
  sum<<<grid_dim, block_dim>>>(A_d, B_d, C_d, BUFSIZE);
  KERNEL_CHECK("launch sum fail");

  // copy result from device to host
  CUDA_CHECK(cudaMemcpy(C_h, C_d, BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost));
  validate(A_h, B_h, C_h, BUFSIZE);
  printf("Run matrix sum successfully\n");

  // free memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
