#include "FrameStreamer.hpp"
#include "Logger.hpp"
#include "PoseEstimator.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

int main( )
{
    std::unique_ptr<Logger::ILogger> logger = std::make_unique<Logger::CoutLogger>( Logger::Priority::Info );

    PoseEstimator model( std::move( logger ) );
    const std::string modelFile = "yolov7-w6-pose.onnx"; // "Yolov5s6_pose_640.onnx";
    model.Initialize(
        std::filesystem::path( __FILE__ ).remove_filename( ).append( modelFile ).wstring( ).c_str( ),
        PoseEstimator::RuntimeBackend::TensorRT,
        "yolo-pose"
    );

    auto RunPoseEstimation = [ &model ]( const cv::Mat& frame ) {
        // * Preprocess data
        const PoseEstimator::InputSize modelInputSize = model.GetModelInputSize( );
        constexpr int batchSize = 1;
        const int size[] = { batchSize, modelInputSize.channels, modelInputSize.width, modelInputSize.height };
        cv::Mat inputBlob( 4, size, CV_32F );
        cv::dnn::blobFromImage(
            frame,
            inputBlob,
            0.00392156862745098,
            cv::Size( modelInputSize.width, modelInputSize.height ),
            cv::Scalar( 0, 0, 0, 0 ),
            true,
            false,
            CV_32F
        );

        // * Forward propagate
        std::vector<PoseEstimator::Detection> output;
        model.Forward( output, ( float* ) inputBlob.data, 640, 640, 3 );

        return FrameStreamer::Result{
            output,
            { .wFactor = static_cast<float>( frame.cols ) / static_cast<float>( modelInputSize.width ),
              .hFactor = static_cast<float>( frame.rows ) / static_cast<float>( modelInputSize.height ) } };
    };

    // const std::string imgFile = "data/img.png";
    // auto fs = CreateFrameStreamer<ImageStreamer>(
    //     std::filesystem::path( __FILE__ ).remove_filename( ).append( imgFile ).string( )
    // );

    const std::string videoFile = "data/dancer.mp4";
    const auto fs = CreateFrameStreamer<VideoStreamer>(
        std::filesystem::path( __FILE__ ).remove_filename( ).append( videoFile ).string( )
    );

    if ( fs )
        fs->Run( RunPoseEstimation );
}

// TODO: Implement "one shot" for just processing a frame
// TODO: Fix find path for onnx
// TODO: Proxy onnxruntime
// TODO: Implement a CameraStreamer
// TODO: Add a frame counter in the image
// TODO: Pimpl to avoid exposing cv::videoio outwards (and ort api from pose estimator)
// TODO: Capture if trying to load image to video streamer
// TODO: Returns silently if cannot find video file
// TODO: Resize video to specified size
// TODO: Separate types to specific headers such as bounding box etc to reduce PoseEstimator dependancies