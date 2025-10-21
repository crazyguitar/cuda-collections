#!/bin/bash

set -exo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sqsh="${DIR}/../../cuda+latest.sqsh"
mount="/fsx:/fsx"
binary="${DIR}/../../build/src/nvshmem-put-nbi-warp/nvshmem-put-nbi-warp"

cmd="$(cat <<EOF
export NVSHMEM_DEBUG=INFO
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NVSHMEM_DISABLE_NVLS=1
${binary}
EOF
)"

srun --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name nvshmem \
  --mpi=pmix \
  --ntasks-per-node=8 \
  bash -c "${cmd}"
