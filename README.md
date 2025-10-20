# CUDA Collections

* [hello](src/hello)
* [sum](src/sum)
* [nvshmem-hello](src/nvshmem-hello)
* [nvshmem-mpi](src/nvshmem-mpi)
* [nvshmem-mpi-uid](src/nvshmem-mpi-uid)
* [nvshmem-put](src/nvshmem-put)
* [nvshmem-put-nbi](src/nvshmem-put-nbi)

```
# build an enroot sqush file
make sqush

# launch an interactive enroot environment
enroot create --name cuda cuda+latest.sqsh
enroot start --mount /fsx:/fsx cuda /bin/bash
```
