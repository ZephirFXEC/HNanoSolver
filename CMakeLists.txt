cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0091 NEW)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /openmp /arch:AVX2") # MSVC

project(HDK_Project LANGUAGES CXX CUDA)

# CMAKE_PREFIX_PATH must contain the path to the toolkit/cmake subdirectory of
# the Houdini installation.
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )

# Locate Houdini's libraries and header files.
find_package(Houdini REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_STANDARD 17)

set(library_name SOP_VdbSolver)

add_library(${library_name} SHARED
        vdb_solver.cpp
        vdb_solver.hpp
        utils.cpp
        utils.hpp
)

target_compile_definitions(${library_name} PRIVATE
        "NANOVDB_USE_OPENVDB"
)


target_link_libraries(${library_name} PRIVATE
        Houdini
        CUDA::cudart
        "C:/Program Files/Side Effects Software/Houdini 20.5.278/custom/houdini/dsolib/openvdb_sesi.lib"
)

target_include_directories(${library_name} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target(${library_name})

# target_sources(${library_name} PRIVATE your_cuda_file.cu)