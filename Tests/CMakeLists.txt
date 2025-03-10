cmake_minimum_required(VERSION 3.28)

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

include_directories(../externals)

add_executable(IndexGrid IndexGrid.cpp)

target_link_libraries(IndexGrid PRIVATE GTest::gtest_main Utils HoudiniVDB Kernels)

# Copy every dll in $ENV{HFS}/bin to the output directory
file(GLOB HFS_DLLS "$ENV{HFS}/bin/*.dll")
file(COPY ${HFS_DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)

# Copy GTest DLLs to the output directory
add_custom_command(TARGET IndexGrid POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:GTest::gtest>
        $<TARGET_FILE:GTest::gtest_main>
        $<TARGET_FILE_DIR:IndexGrid>)

include(GoogleTest)
gtest_discover_tests(IndexGrid)
