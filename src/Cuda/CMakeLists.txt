enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")

add_library(Kernels STATIC
        PointToGrid.cuh
        utils.cuh
        HNanoGrid/HNanoGrid.cuh

        Advection.cu
        PressureProjection.cu
        PointToGrid.cu
)

target_compile_features(Kernels PUBLIC cxx_std_17)
target_link_libraries(Kernels CUDA::cudart)