cmake_minimum_required(VERSION 3.29)

cmake_policy(SET CMP0135 NEW)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(BUILD_GMOCK OFF CACHE BOOL "Whether or not to build gmock" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib?" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(IndexGrid IndexGrid.cpp)
target_link_libraries(IndexGrid PRIVATE GTest::gtest_main Utils HoudiniVDB Kernels)

include(GoogleTest)
gtest_discover_tests(IndexGrid)
