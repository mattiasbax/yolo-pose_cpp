project(YoloPoseCpp)
cmake_minimum_required(VERSION 3.23)

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h)
find_library(ONNX_RUNTIME_LIB onnxruntime)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)

add_executable(yolo_pose_cpp main.cpp FrameStreamer.hpp PoseEstimator.hpp)
set_target_properties(yolo_pose_cpp PROPERTIES CXX_STANDARD 20)

target_include_directories(yolo_pose_cpp PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})
target_link_libraries(yolo_pose_cpp PRIVATE opencv_core opencv_dnn ${OpenCV_LIBRARIES} ${OpenCV_LIBRARIES} ${ONNX_RUNTIME_LIB})
