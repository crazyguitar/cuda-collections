file(GLOB src *.cu)
set(target p2p-nvlink)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${target} ${src})
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target} PRIVATE CUDA::cudart)
