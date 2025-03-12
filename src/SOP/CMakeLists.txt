cmake_minimum_required(VERSION 3.28)
project(HNanoSOP LANGUAGES CUDA CXX)

houdini_generate_proto_headers(FILES
        HNanoSolver/SOP_HNanoSolver.cpp
        Advection/SOP_VDBAdvect.cpp
        ReadWrite/SOP_VDBFromGrid.cpp
        VelocityAdvection/SOP_VDBAdvectVelocity.cpp
        ProjectNonDivergent/SOP_VDBProjectNonDivergent.cpp
)

include_directories(../../externals)

add_subdirectory(../Cuda Kernels)
########################  Utils library ################################################################################
target_include_directories(Utils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_features(Utils PUBLIC cxx_std_17)
########################################################################################################################


########################  SOP library ##################################################################################

function(add_sop_library LIBNAME SRC_FILES)
    add_library(${LIBNAME} SHARED ${SRC_FILES})
    target_link_libraries(${LIBNAME} PRIVATE Utils Kernels HoudiniVDB BrickMap)
    target_include_directories(${LIBNAME} PRIVATE
            .
            ${CMAKE_CURRENT_BINARY_DIR}
            $ENV{HFS}/toolkit/include
    )
    target_compile_features(${LIBNAME} PUBLIC cxx_std_17)
endfunction()

add_sop_library(SOP_HNanoSolver
        "HNanoSolver/SOP_HNanoSolver.cpp"
        "HNanoSolver/SOP_HNanoSolver.hpp"
)

add_sop_library(SOP_Advect
        "Advection/SOP_VDBAdvect.cpp"
        "Advection/SOP_VDBAdvect.hpp"
)

add_sop_library(SOP_ProjectNonDivergent
        "ProjectNonDivergent/SOP_VDBProjectNonDivergent.cpp"
        "ProjectNonDivergent/SOP_VDBProjectNonDivergent.hpp"
)

add_sop_library(SOP_AdvectVelocity
        "VelocityAdvection/SOP_VDBAdvectVelocity.cpp"
        "VelocityAdvection/SOP_VDBAdvectVelocity.hpp"
)

add_sop_library(SOP_VDBFromGrid
        "ReadWrite/SOP_VDBFromGrid.cpp"
        "ReadWrite/SOP_VDBFromGrid.hpp"
)

########################################################################################################################

houdini_configure_target(SOP_VDBFromGrid $ENV{DSO_Path})
houdini_configure_target(SOP_HNanoSolver $ENV{DSO_Path})
houdini_configure_target(SOP_Advect $ENV{DSO_Path})
houdini_configure_target(SOP_AdvectVelocity $ENV{DSO_Path})
houdini_configure_target(SOP_ProjectNonDivergent $ENV{DSO_Path})