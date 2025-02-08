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
target_link_directories(IndexGrid PRIVATE "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.42.34321/lib/x64")

#target_compile_options(IndexGrid PRIVATE $<$<CONFIG:Debug,RelWithDebInfo>:/fsanitize=address>)
target_link_libraries(IndexGrid PRIVATE GTest::gtest_main Utils HoudiniVDB Kernels)

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    get_filename_component(CL_DIR ${CMAKE_C_COMPILER} DIRECTORY)

    set(ASAN_COPY_COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${CL_DIR}/clang_rt.asan_dynamic-x86_64.dll $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

    add_custom_command(
            TARGET ${TESTNAME} POST_BUILD
            COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${ASAN_COPY_COMMAND}>"
            COMMAND_EXPAND_LISTS
    )

endif()

include(GoogleTest)
gtest_discover_tests(IndexGrid)
