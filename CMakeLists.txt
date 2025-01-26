cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(BUILD_SHARED_LIBS ON)

project(HNanoSolver LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)


# ------- Find Houdini --------
# Points CMake to $HFS/toolkit/cmake
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

# Optionally create an interface library capturing Houdini & openvdb_sesi
add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
target_include_directories(HoudiniVDB INTERFACE
        $ENV{HFS}/toolkit/include
)

houdini_generate_proto_headers( FILES
        #src/SOP/HNanoSolver/SOP_HNanoSolver.cpp
        src/SOP/Advection/SOP_VDBAdvect.cpp
        src/SOP/ReadWrite/SOP_VDBFromGrid.cpp
        src/SOP/VelocityAdvection/SOP_VDBAdvectVelocity.cpp
        src/SOP/ProjectNonDivergent/SOP_VDBProjectNonDivergent.cpp
)


# Add subdirectories
add_subdirectory(src/Cuda)
add_subdirectory(src/Tests)
add_subdirectory(src/HNanoViewer)

########################  Utils library ################################################################################

add_library(Utils STATIC
        src/Utils/Utils.cpp
        src/Utils/ScopedTimer.hpp
        src/Utils/OpenToNano.hpp
        src/Utils/GridData.hpp
        src/Utils/Memory.hpp
        src/Utils/Utils.hpp
)
target_link_libraries(Utils PRIVATE HoudiniVDB)

target_include_directories(Utils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_features(Utils PUBLIC cxx_std_17)
########################################################################################################################


########################  SOP library ##################################################################################

function(add_sop_library LIBNAME SRC_FILES)
    add_library(${LIBNAME} SHARED ${SRC_FILES})
    target_link_libraries(${LIBNAME} PRIVATE Utils Kernels HoudiniVDB)
    target_include_directories(${LIBNAME} PRIVATE
            src
            ${CMAKE_CURRENT_BINARY_DIR}
            $ENV{HFS}/toolkit/include
    )
    target_compile_features(${LIBNAME} PUBLIC cxx_std_17)
endfunction()

add_sop_library(SOP_HNanoSolver
        "src/SOP/HNanoSolver/SOP_HNanoSolver.cpp"
        "src/SOP/HNanoSolver/SOP_HNanoSolver.hpp"
)

add_sop_library(SOP_Advect
        "src/SOP/Advection/SOP_VDBAdvect.cpp"
        "src/SOP/Advection/SOP_VDBAdvect.hpp"
)

add_sop_library(SOP_ProjectNonDivergent
        "src/SOP/ProjectNonDivergent/MultigridSolver.hpp"
        "src/SOP/ProjectNonDivergent/SOP_VDBProjectNonDivergent.cpp"
        "src/SOP/ProjectNonDivergent/SOP_VDBProjectNonDivergent.hpp"
)

add_sop_library(SOP_AdvectVelocity
        "src/SOP/VelocityAdvection/SOP_VDBAdvectVelocity.cpp"
        "src/SOP/VelocityAdvection/SOP_VDBAdvectVelocity.hpp"
)

add_sop_library(SOP_VDBFromGrid
        "src/SOP/ReadWrite/SOP_VDBFromGrid.cpp"
        "src/SOP/ReadWrite/SOP_VDBFromGrid.hpp"
)

########################################################################################################################

# houdini_configure_target(SOP_HNanoSolver $ENV{DSO_Path})
houdini_configure_target(SOP_VDBFromGrid $ENV{DSO_Path})
houdini_configure_target(SOP_Advect $ENV{DSO_Path})
houdini_configure_target(SOP_AdvectVelocity $ENV{DSO_Path})
houdini_configure_target(SOP_ProjectNonDivergent $ENV{DSO_Path})

add_custom_target(run_tests ALL
        COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
        DEPENDS IndexGrid
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Running unit tests after build..."
)