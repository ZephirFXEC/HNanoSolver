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

add_subdirectory(src/Cuda)
# ------- End CUDA -----------

# ------- Find Houdini -----------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )
find_package(Houdini REQUIRED)
# ------- End Houdini -----------

houdini_generate_proto_headers( FILES
        src/SOP/Advection/SOP_VDBAdvect.cpp
        src/SOP/ReadWrite/SOP_VDBFromGrid.cpp
        src/SOP/VelocityAdvection/SOP_VDBAdvectVelocity.cpp
        src/SOP/ProjectNonDivergent/SOP_VDBProjectNonDivergent.cpp
)

########################  Utils library ########################################################################
add_library(Utils STATIC
        src/Utils/Utils.cpp
        src/Utils/Utils.hpp
        src/Utils/ScopedTimer.hpp
        src/Utils/OpenToNano.hpp
        src/Utils/GridData.hpp
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


########################  SOP_Advect library ###########################################################################
add_library(SOP_Advect SHARED
        src/SOP/Advection/SOP_VDBAdvect.cpp
        src/SOP/Advection/SOP_VDBAdvect.hpp
)
target_link_libraries(SOP_Advect PRIVATE
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(SOP_Advect PRIVATE
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################


########################  SOP_ProjectNonDivergent library ##############################################################
add_library(SOP_ProjectNonDivergent SHARED
        src/SOP/ProjectNonDivergent/SOP_VDBProjectNonDivergent.cpp
        src/SOP/ProjectNonDivergent/SOP_VDBProjectNonDivergent.hpp
)
target_link_libraries(SOP_ProjectNonDivergent PRIVATE
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(SOP_ProjectNonDivergent PRIVATE
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################


########################  SOP_AdvectVelocity library ###################################################################
add_library(SOP_AdvectVelocity SHARED
        src/SOP/VelocityAdvection/SOP_VDBAdvectVelocity.cpp
        src/SOP/VelocityAdvection/SOP_VDBAdvectVelocity.hpp
)
target_link_libraries(SOP_AdvectVelocity PRIVATE
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(SOP_AdvectVelocity PRIVATE
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################


########################  SOP_ReadWriteTest library ####################################################################
add_library(SOP_VDBFromGrid SHARED
        src/SOP/ReadWrite/SOP_VDBFromGrid.cpp
        src/SOP/ReadWrite/SOP_VDBFromGrid.hpp
)
target_link_libraries(SOP_VDBFromGrid PRIVATE
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib"
)
target_include_directories(SOP_VDBFromGrid PRIVATE
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################
houdini_configure_target(SOP_VDBFromGrid $ENV{DSO_Path})
houdini_configure_target(SOP_Advect $ENV{DSO_Path})
houdini_configure_target(SOP_AdvectVelocity $ENV{DSO_Path})
houdini_configure_target(SOP_ProjectNonDivergent $ENV{DSO_Path})
