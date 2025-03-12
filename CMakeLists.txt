cmake_minimum_required(VERSION 3.28)
project(HNanoSolver LANGUAGES CXX CUDA VERSION 1.0.0)

# Enforce C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# Set vars
if (UNIX)
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
endif ()

if (WIN32)
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
endif ()


# Define build options
option(HNS_BUILD_SOP "Build SOP components" ON)
option(HNS_BUILD_TESTS "Build tests" OFF)
option(HNS_BUILD_VIEWER "Build HNanoViewer" ON)

# ------- Find Houdini --------
# Points CMake to $HFS/toolkit/cmake
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

# Ccreate an interface library capturing Houdini & openvdb_sesi
add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})
target_include_directories(HoudiniVDB INTERFACE
        $ENV{HFS}/toolkit/include
)

add_library(Utils STATIC
        src/Utils/ScopedTimer.hpp
        src/Utils/GridBuilder.hpp
        src/Utils/GridData.hpp
        src/Utils/Memory.hpp
        src/Utils/Utils.hpp
        src/Utils/Utils.cpp
        src/Utils/Stencils.hpp
)

target_link_libraries(Utils PRIVATE HoudiniVDB)

# Set default build type if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Component builds
if (HNS_BUILD_SOP)
    add_subdirectory(src/SOP)
endif ()

if (HNS_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_subdirectory(Tests)

    add_custom_target(run_tests ALL
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            DEPENDS IndexGrid
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running unit tests after build..."
    )
endif ()

if (HNS_BUILD_VIEWER)
    add_subdirectory(HNanoViewer)
endif ()