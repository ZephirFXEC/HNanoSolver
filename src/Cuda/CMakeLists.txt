enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")
include_directories(../../externals)

add_library(Kernels STATIC
        Utils.cuh
        Kernels.cu
        Advection.cu
        PressureProjection.cu
)

target_compile_options(Kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_include_directories(Kernels
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        .  # for your .cuh headers
)
target_link_libraries(Kernels PRIVATE CUDA::cudart)

target_compile_features(Kernels PUBLIC cxx_std_17)