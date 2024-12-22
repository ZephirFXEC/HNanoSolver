cmake_minimum_required(VERSION 3.29)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_CXX_STANDARD 17)

project(HNanoSolver LANGUAGES CXX)

add_compile_definitions(
        SESI_OPENVDB
)

# ------- Find Houdini -----------
list(APPEND CMAKE_PREFIX_PATH "/Applications/Houdini/Houdini20.5.445/Frameworks/Houdini.framework/Versions/20.5/Resources/toolkit/cmake/")
find_package(Houdini REQUIRED)
# ------- End Houdini -----------

########################  Utils library ########################################################################
add_library(Utils STATIC
        src/Utils/ScopedTimer.hpp
        src/Utils/OpenToNano.hpp
        src/Utils/GridData.hpp
        src/Utils/Memory.hpp
        src/Utils/Utils.cpp
        src/Utils/Utils.hpp
)
target_link_libraries(Utils PRIVATE
        Houdini
        "/Applications/Houdini/Houdini20.5.445/Frameworks/Houdini.framework/Libraries/libopenvdb_sesi.11.0.dylib")

target_include_directories(Utils PUBLIC
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        $ENV{HFS}/toolkit/include
)
########################################################################################################################

# ------- Compile HNanoViewer -
add_subdirectory(src/HNanoViewer)
# ------- End HNanoViewer -----