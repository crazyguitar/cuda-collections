# CUDA Collections

* [hello](src/hello)

```
# build an enroot sqush file
make sqush

# launch an interactive enroot environment
enroot create --name cuda cuda+latest.sqsh
enroot start --mount /fsx:/fsx cuda /bin/bash
```
