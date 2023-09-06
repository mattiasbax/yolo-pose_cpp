#include "FrameStreamer.hpp"
#include "Logger.hpp"
#include "PoseEstimator.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

int main( )
{
    std::unique_ptr<Logger::ILogger> logger = std::make_unique<Logger::CoutLogger>( Logger::Priority::Info );

    PoseEstimator model( std::move( logger ) );
    const std::string modelFile = "yolov7-w6-pose.onnx"; // "Yolov5s6_pose_640.onnx"; // "yolov7-w6-pose.onnx";
    model.Initialize(
        std::filesystem::path( __FILE__ ).remove_filename( ).append( modelFile ).wstring( ).c_str( ),
        PoseEstimator::RuntimeBackend::TensorRT,
        "yolo-pose"
    );

    auto RunPoseEstimation = [ &model ]( const cv::Mat& frame ) {
        const PoseEstimator::InputSize modelInputSize = model.GetModelInputSize( );

        const double confidenceThreshold = 0.3;

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

        std::vector<PoseEstimator::Detection> output;
        model.Forward( output, ( float* ) inputBlob.data, 640, 640, 3 );

        const float wFactor = static_cast<float>( frame.cols ) / static_cast<float>( modelInputSize.width );
        const float hFactor = static_cast<float>( frame.rows ) / static_cast<float>( modelInputSize.height );

        for ( const auto& detection : output ) {
            const cv::Scalar colorBox = { 200, 0, 0 };      // blue
            const cv::Scalar colorSkeleton = { 0, 200, 0 }; // green
            const cv::Scalar colorJoints = { 0, 0, 200 };   // red
            if ( detection.box.score < confidenceThreshold )
                continue;
            const cv::Point2f tl = { detection.box.tlX * wFactor, detection.box.tlY * hFactor };
            const cv::Point2f br = { detection.box.brX * wFactor, detection.box.brY * hFactor };
            cv::rectangle( frame, tl, br, colorBox, 2 );

            for ( int i = 0; i < detection.keyPoints.size( ); ++i ) {
                const auto& keypoint = detection.keyPoints[ i ];
                if ( keypoint.score < confidenceThreshold )
                    continue;

                const cv::Point2f center = { keypoint.x * wFactor, keypoint.y * hFactor };
                cv::circle( frame, center, 3, colorJoints, -1 );
            }

            for ( const auto& edge : PoseEstimator::skeleton ) {
                if ( detection.keyPoints[ edge.first ].score < confidenceThreshold
                     || detection.keyPoints[ edge.second ].score < confidenceThreshold )
                    continue;
                cv::Point2f from = {
                    detection.keyPoints[ edge.first ].x * wFactor, detection.keyPoints[ edge.first ].y * hFactor };
                cv::Point2f to = {
                    detection.keyPoints[ edge.second ].x * wFactor, detection.keyPoints[ edge.second ].y * hFactor };
                cv::line( frame, from, to, colorSkeleton );
            }
        }
    };

    const std::string imgFile = "data/img.png";
    const std::string videoFile = "data/dancer.mp4";

    // auto fs = CreateFrameStreamer<ImageStreamer>(
    //     std::filesystem::path( __FILE__ ).remove_filename( ).append( imgFile ).string( ), 100
    // );
    const auto fs = CreateFrameStreamer<VideoStreamer>(
        std::filesystem::path( __FILE__ ).remove_filename( ).append( videoFile ).string( ), 150
    );

    if ( fs )
        fs->Run( RunPoseEstimation );
}

// TODO: Fix find path for onnx
// TODO: Proxy onnxruntime
// TODO: Implement a CameraStreamer
// TODO: Add a frame counter in the image
// TODO: Pimpl to avoid exposing cv::videoio outwards (and ort api from pose estimator)
// TODO: Capture if trying to load image to video streamer
// TODO: Returns silently if cannot find video file
// TODO: Resize video to specified size