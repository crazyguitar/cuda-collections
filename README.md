# CUDA Collections

* [hello](src/hello)
* [sum](src/sum)
* [warp](src/warp)
* [\_\_shfl](src/__shfl)
* [device-symbol](src/device-symbol)
* [zero-copy](src/zero-copy)
* [p2p-ipc](src/p2p-ipc)
* [p2p-nvlink](src/p2p-nvlink)
* [nvshmem-hello](src/nvshmem-hello)
* [nvshmem-mpi](src/nvshmem-mpi)
* [nvshmem-mpi-uid](src/nvshmem-mpi-uid)
* [nvshmem-put](src/nvshmem-put)
* [nvshmem-put-nbi](src/nvshmem-put-nbi)
* [nvshmem-put-nbi-block](src/nvshmem-put-nbi-block)
* [nvshmem-put-nbi-warp](src/nvshmem-put-nbi-warp)
* [nvshmem-collective-launch](src/nvshmem-collective-launch)
* [nvshmem-all2all](src/nvshmem-all2all)
* [nvshmem-moe-count](src/nvshmem-moe-count)
* [nccl-symmetric-memory ](src/nccl-symmetric-memory)

```
# build an enroot sqush file
make sqush

# launch an interactive enroot environment
enroot create --name cuda cuda+latest.sqsh
enroot start --mount /fsx:/fsx cuda /bin/bash
```
