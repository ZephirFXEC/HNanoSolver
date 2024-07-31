cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_CXX_STANDARD 17)

project(HDK_Project LANGUAGES CXX CUDA)

# ------- Find Houdini -----------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )
find_package(Houdini REQUIRED)
# ------- End Houdini -----------


# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_STANDARD 17)
# ------- End CUDA -----------

set(library_name SOP_VdbSolver)

add_library(${library_name} SHARED
        src/SOP_VDBSolver.cpp
        src/SOP_VDBSolver.hpp
        src/OpenVDB_Utils/ParmFactory.cpp
        src/OpenVDB_Utils/ParmFactory.hpp
        src/OpenVDB_Utils/SOP_NodeVDB.cpp
        src/OpenVDB_Utils/SOP_NodeVDB.hpp
        src/OpenVDB_Utils/SOP_VDBVerbUtils.hpp
        src/OpenVDB_Utils/Geometry.cpp
        src/OpenVDB_Utils/Geometry.hpp
        src/OpenVDB_Utils/PointUtils.cpp
        src/OpenVDB_Utils/PointUtils.hpp
        src/OpenVDB_Utils/Utils.cpp
        src/OpenVDB_Utils/Utils.hpp
        src/OpenVDB_Utils/AttributeTransferUtil.hpp
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