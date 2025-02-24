cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(BUILD_SHARED_LIBS ON)

project(HNanoSolver LANGUAGES CXX CUDA)

option(SOP "src/SOP" ON)
option(Tests "Test" OFF)
option(HNanoViewer "HNanoViewer" OFF)

if (SOP)
    add_subdirectory(src/SOP)
endif ()

if (Tests)
    include(CTest)
    enable_testing()

    add_subdirectory(Tests)

    add_custom_target(run_tests ALL
            COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
            DEPENDS IndexGrid
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Running unit tests after build..."
    )
endif ()

if (HNanoViewer)
    add_subdirectory(HNanoViewer)
endif ()
