cmake_minimum_required(VERSION 3.29)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_CXX_STANDARD 17)

project(HNanoSolver LANGUAGES CXX CUDA)

add_compile_definitions(
        SESI_OPENVDB
        NANOVDB_USE_OPENVDB
)


# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda")
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(BUILD_SHARED_LIBS ON)

add_subdirectory(src/Kernel)
# ------- End CUDA -----------

# ------- Find Houdini -----------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )
find_package(Houdini REQUIRED)
# ------- End Houdini -----------


add_library(${PROJECT_NAME} SHARED
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
        src/Utils/IncomingDataCache.hpp
        src/Utils/ScopedTimer.hpp
        src/Utils/CudaStreamWrapper.hpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Kernels
        Houdini
       "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target(${PROJECT_NAME})
