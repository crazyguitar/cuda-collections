#include <cuda.h>

#include <cstdio>

__global__ void hello() { printf("thread: %d, block: %d\n", threadIdx.x, blockIdx.x); }

int main(int argc, char* argv[]) {
  hello<<<2, 2>>>();
  cudaDeviceSynchronize();
}
