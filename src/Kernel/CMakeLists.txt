enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")

add_library(Kernels STATIC
        thrust_kernel.cu
        pointsToGrid_kernel.cu
        utils.cuh
)

target_compile_features(Kernels PUBLIC cxx_std_17)
target_link_libraries(Kernels CUDA::cudart)