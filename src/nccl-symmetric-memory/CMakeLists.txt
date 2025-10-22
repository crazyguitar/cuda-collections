file(GLOB src *.cc *.cu *.cpp)
set(target nccl-symmetric-memory)
set(ENV{PKG_CONFIG_PATH} "/opt/amazon/efa/lib/pkgconfig")
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${target} ${src})
set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_include_directories(${target} PRIVATE
  "/opt/nccl/build/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${MPI_INCLUDE_PATH}"
)

target_link_libraries(${target} PRIVATE
  Threads::Threads
  "${MPI_LIBRARIES}"
  CUDA::cudart
  nccl
)
