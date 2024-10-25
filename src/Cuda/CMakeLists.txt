enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")

add_library(Kernels STATIC
        Advection.cu
        PointToGrid.cu
        utils.cuh
        PressureProjection.cu
        HNanoGrid/HNanoGrid.cuh
)

target_compile_features(Kernels PUBLIC cxx_std_17)
target_link_libraries(Kernels CUDA::cudart)