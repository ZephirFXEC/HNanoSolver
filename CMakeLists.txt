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

houdini_generate_proto_headers( FILES
        src/SOP/Solver/SOP_VDBSolver.cpp
        src/SOP/ReadWrite/SOP_ReadWriteTest.cpp
)

########################  Utils library ########################################################################
add_library(Utils STATIC
        src/Utils/Utils.cpp
        src/Utils/Utils.hpp
        src/Utils/IncomingDataCache.hpp
        src/Utils/ScopedTimer.hpp
        src/Utils/CudaStreamWrapper.hpp
)
target_link_libraries(Utils PRIVATE
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(Utils PUBLIC
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################


########################  SOP_VDBSolver library ########################################################################
add_library(SOP_VDBSolver SHARED
        src/SOP/Solver/SOP_VDBSolver.cpp
        src/SOP/Solver/SOP_VDBSolver.hpp
)
target_link_libraries(SOP_VDBSolver PRIVATE
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(SOP_VDBSolver PRIVATE
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################


########################  SOP_ReadWriteTest library ####################################################################
add_library(SOP_ReadWriteTest SHARED
        src/SOP/ReadWrite/SOP_ReadWriteTest.cpp
        src/SOP/ReadWrite/SOP_ReadWriteTest.hpp
)
target_link_libraries(SOP_ReadWriteTest PRIVATE
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(SOP_ReadWriteTest PRIVATE
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################
houdini_configure_target(SOP_ReadWriteTest $ENV{DSO_Path})
houdini_configure_target(SOP_VDBSolver $ENV{DSO_Path})
