enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")
include_directories(../../externals)

add_library(BrickMap STATIC
        BrickMap.cuh
        BrickMap.cu
        Sampler.cuh
)

target_compile_options(BrickMap PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_include_directories(BrickMap
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}  # for your .cuh headers
)
target_link_libraries(BrickMap PRIVATE CUDA::cudart)

target_compile_features(BrickMap PUBLIC cxx_std_17)