cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(HNanoViewer LANGUAGES CXX CUDA)

# Find packages
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_executable(HNanoViewer src/HNanoViewer.cpp src/OpenVDBLoader.hpp src/OpenVDBLoader.cpp src/Shader.hpp src/Shader.cpp src/Renderer.hpp src/Renderer.cpp)

target_link_libraries(HNanoViewer PRIVATE
        glad::glad
        glfw
        imgui::imgui
        Utils
        Kernels
        Houdini
        "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")