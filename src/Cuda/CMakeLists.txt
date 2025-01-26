enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")

add_library(Kernels STATIC
        PointToGrid.cuh
        utils.cuh
        Kernels.cu
        Advection.cu
        PressureProjection.cu
        PointToGrid.cu
)

target_compile_options(Kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_include_directories(Kernels
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}  # for your .cuh headers
)
target_link_libraries(Kernels PRIVATE CUDA::cudart)

target_compile_features(Kernels PUBLIC cxx_std_17)