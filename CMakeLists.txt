cmake_minimum_required(VERSION 3.6)

project(HDK_Project)

# CMAKE_PREFIX_PATH must contain the path to the toolkit/cmake subdirectory of
# the Houdini installation. See the "Compiling with CMake" section of the HDK
# documentation for more details, which describes several options for
# specifying this path.
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package(Houdini REQUIRED)

set(library_name SOP_VdbSolver)


# Add a library and its source files.
add_library(${library_name} SHARED
        vdb_solver.cpp
        vdb_solver.hpp
        utils.cc
        utils.h
)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries(${library_name} Houdini)
target_link_libraries(${library_name} "C:/Program Files/Side Effects Software/Houdini 20.5.278/custom/houdini/dsolib/openvdb_sesi.lib")
target_link_libraries(${library_name} "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64/cudart.lib")

# Include ${CMAKE_CURRENT_BINARY_DIR} for the generated header.
target_include_directories(${library_name} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(${library_name} PRIVATE
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"
)


# Sets several common target properties, such as the library's output directory.
houdini_configure_target(${library_name})
