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

# Define build options
option(HNS_BUILD_SOP "Build SOP components" ON)
option(HNS_BUILD_TESTS "Build tests" ON)
option(HNS_BUILD_VIEWER "Build HNanoViewer" OFF)

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