include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
) 
FetchContent_MakeAvailable(googletest)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(yolo_pose_cpp_tests test_main.cpp)

# target_include_directories(yolo_pose_cpp_tests PRIVATE)

target_link_libraries(yolo_pose_cpp_tests PRIVATE
    gtest_main
    gmock_main)