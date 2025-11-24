# GPU/CPU Zero Copy

Transferring data between GPU and CPU using `cudaMemcpy` can be expensive. For applications with frequent data transfers, performance may suffer. CUDA provides three APIs to optimize memory access: `cudaMallocManaged`, `cudaMallocHost`, and `cudaHostRegister`.

## cudaMallocManaged
* Allocates unified memory accessible by both GPU and CPU
* Uses a single pointer with automatic data migration
* System automatically moves data between GPU and CPU as needed
* Simplifies programming but involves physical memory transfers

## cudaMallocHost
* Allocates pinned (page-locked) memory on the host
* Enables faster DMA transfers with explicit `cudaMemcpy`
* When combined with `cudaHostGetDevicePointer` (on supported systems):
  * Enables true zero-copy access
  * GPU reads directly from host memory over PCIe
  * No data migration - memory stays on host
  * Requires unified addressing support

## cudaHostRegister
* Pins existing host memory (from `malloc`/`new`)
* Makes pre-allocated memory DMA-friendly
* Useful when allocation is controlled by external code
* When combined with `cudaHostGetDevicePointer` (on supported systems):
  * Enables zero-copy for existing allocations
  * GPU accesses host memory directly over PCIe
