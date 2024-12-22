# src/HNanoViewer/CMakeLists.txt

cmake_minimum_required(VERSION 3.29)
project(HNanoViewer LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Create GLAD library with the proper path
add_library(glad STATIC
        ${CMAKE_CURRENT_LIST_DIR}/glad/src/glad.c
)
target_include_directories(glad PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/glad/include
)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)


# 2) Find needed packages
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# 3) Add your executable
add_executable(HNanoViewer
        src/HNanoViewer.cpp
        src/OpenVDBLoader.cpp
        src/OpenVDBLoader.hpp
        src/Shader.cpp
        src/Shader.hpp
        src/Renderer.cpp
        src/Renderer.hpp
)

# 4) Link everything
target_link_libraries(HNanoViewer PRIVATE
        glad
        glfw
        imgui::imgui
        glm::glm
        Utils
        Houdini
        "/Applications/Houdini/Houdini20.5.445/Frameworks/Houdini.framework/Libraries/libopenvdb_sesi.11.0.dylib"
)
