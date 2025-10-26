#!/bin/bash

set -ex

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
binary="${DIR}/../../build/src/nvshmem-moe/nvshmem-moe"

/opt/hydra/bin/nvshmrun \
  -ppn 8  \
  --bind-to none \
  --genv NVSHMEM_DEBUG=WARN \
  --genv NVSHMEM_REMOTE_TRANSPORT=libfabric \
  --genv NVSHMEM_LIBFABRIC_PROVIDER=efa \
  "${binary}"
