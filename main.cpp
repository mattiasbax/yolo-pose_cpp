#include "FrameStreamer.hpp"
#include "PoseEstimator.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main( )
{
    std::filesystem::path modelPath = __FILE__;
    modelPath.remove_filename( ).append( "yolov7-w6-pose.onnx" );
    // modelPath.remove_filename( ).append( "Yolov5s6_pose_640.onnx" );

    PoseEstimator model;
    model.Initialize( modelPath.wstring( ).c_str( ), "yolo-pose" );

    std::filesystem::path imgPath = __FILE__;
    imgPath.remove_filename( ).append( "data/img.png" );
    cv::Mat inputImage = cv::imread( imgPath.string( ) );
    cv::resize( inputImage, inputImage, cv::Size( 640, 640 ) ); // resize to network image size

    std::filesystem::path videoPath = __FILE__;
    videoPath.remove_filename( ).append( "data/dancer.mp4" );

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

    auto fs = CreateFrameStreamer<ImageStreamer>( imgPath.string( ), 100 );

    if ( fs )
        fs->Run( RunPoseEstimation );
}