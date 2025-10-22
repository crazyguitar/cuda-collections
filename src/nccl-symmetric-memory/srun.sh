#!/bin/bash

set -exo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sqsh="${DIR}/../../cuda+latest.sqsh"
mount="/fsx:/fsx"
binary="${DIR}/../../build/src/nccl-symmetric-memory/nccl-symmetric-memory"

cmd="$(cat <<EOF
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so
export NCCL_NVLS_ENABLE=0
${binary}
EOF
)"

srun --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name nvshmem \
  --mpi=pmix \
  --ntasks-per-node=8 \
  bash -c "${cmd}"
