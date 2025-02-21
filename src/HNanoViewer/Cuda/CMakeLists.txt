enable_language(CUDA)

include_directories("$ENV{HFS}/toolkit/include")

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20")

find_package(OpenGL REQUIRED)  # Add this

add_library(HNanoViewerCuda STATIC
        volumeTex.cu
        VolumeTex.cuh
)

include_directories(../../../src/Cuda/BrickMap)

target_compile_options(HNanoViewerCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_include_directories(HNanoViewerCuda
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GLAD_INCLUDE_DIR}  # Add path to glad.h
        ${OpenGL_INCLUDE_DIRS}
)
target_link_libraries(HNanoViewerCuda PRIVATE
        CUDA::cudart
        OpenGL::GL  # Add OpenGL
        glad::glad  # Add GLAD
)

target_compile_features(HNanoViewerCuda PUBLIC cxx_std_17)