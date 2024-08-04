cmake_minimum_required(VERSION 3.26)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)

project(HNanoSolver LANGUAGES CXX CUDA)

# ------- Find Houdini -----------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )
find_package(Houdini REQUIRED)
# ------- End Houdini -----------

# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
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

target_link_libraries(${library_name} PRIVATE
        Houdini
        CUDA::cudart
)

target_include_directories(${library_name} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        $ENV{HFS}/toolkit/include
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target(${library_name})

# target_sources(${library_name} PRIVATE your_cuda_file.cu)