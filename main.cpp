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

    YoloV7Pose model;
    model.Initialize( modelPath.wstring( ).c_str( ), "yolo-pose" );

    std::filesystem::path imgPath = __FILE__;
    imgPath.remove_filename( ).append( "img.png" );
    cv::Mat inputImage = cv::imread( imgPath.string( ) );
    cv::resize( inputImage, inputImage, cv::Size( 640, 640 ) ); // resize to network image size

    const std::vector<std::pair<YoloV7Pose::Joint, YoloV7Pose::Joint>> skeleton{
        { YoloV7Pose::Joint::leftAnkle, YoloV7Pose::Joint::leftKnee },
        { YoloV7Pose::Joint::leftKnee, YoloV7Pose::Joint::leftHip },
        { YoloV7Pose::Joint::rightAnkle, YoloV7Pose::Joint::rightKnee },
        { YoloV7Pose::Joint::rightKnee, YoloV7Pose::Joint::rightHip },
        { YoloV7Pose::Joint::leftHip, YoloV7Pose::Joint::rightHip },
        { YoloV7Pose::Joint::leftShoulder, YoloV7Pose::Joint::leftHip },
        { YoloV7Pose::Joint::rightShoulder, YoloV7Pose::Joint::rightHip },
        { YoloV7Pose::Joint::leftShoulder, YoloV7Pose::Joint::rightShoulder },
        { YoloV7Pose::Joint::leftShoulder, YoloV7Pose::Joint::leftElbow },
        { YoloV7Pose::Joint::rightShoulder, YoloV7Pose::Joint::rightElbow },
        { YoloV7Pose::Joint::leftElbow, YoloV7Pose::Joint::leftWrist },
        { YoloV7Pose::Joint::rightElbow, YoloV7Pose::Joint::rightWrist },
        { YoloV7Pose::Joint::leftEye, YoloV7Pose::Joint::rightEye },
        { YoloV7Pose::Joint::Nose, YoloV7Pose::Joint::leftEye },
        { YoloV7Pose::Joint::Nose, YoloV7Pose::Joint::rightEye },
        { YoloV7Pose::Joint::leftEye, YoloV7Pose::Joint::leftEar },
        { YoloV7Pose::Joint::rightEye, YoloV7Pose::Joint::rightEar },
        { YoloV7Pose::Joint::leftEar, YoloV7Pose::Joint::leftShoulder },
        { YoloV7Pose::Joint::rightEar, YoloV7Pose::Joint::rightShoulder } };

    std::filesystem::path videoPath = __FILE__;
    videoPath.remove_filename( ).append( "video.mp4" );

    auto RunPoseEstimation = [ &model, &skeleton ]( const cv::Mat& frame ) {
        const int networkInputWidth = 640;
        const int networkInputHeight = 640;

        const double confidenceThreshold = 0.3;

        const int size[] = { 1, 3, networkInputWidth, networkInputHeight };
        cv::Mat inputBlob( 4, size, CV_32F );
        cv::dnn::blobFromImage( frame,
                                inputBlob,
                                0.00392156862745098,
                                cv::Size( 640, 640 ),
                                cv::Scalar( 0, 0, 0, 0 ),
                                true,
                                false,
                                CV_32F );

        std::vector<YoloV7Pose::Detection> output;
        model.Forward( output, (float*) inputBlob.data, 640, 640, 3 );

        const int frameWidth = frame.cols;
        const int frameHeight = frame.rows;

        const float wFactor = static_cast<float>( frameWidth ) / static_cast<float>( networkInputWidth );
        const float hFactor = static_cast<float>( frameHeight ) / static_cast<float>( networkInputHeight );

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

            for ( const auto& edge : skeleton ) {
                if ( detection.keyPoints[ edge.first ].score < confidenceThreshold
                     || detection.keyPoints[ edge.second ].score < confidenceThreshold )
                    continue;
                cv::Point2f from = { detection.keyPoints[ edge.first ].x * wFactor,
                                     detection.keyPoints[ edge.first ].y * hFactor };
                cv::Point2f to = { detection.keyPoints[ edge.second ].x * wFactor,
                                   detection.keyPoints[ edge.second ].y * hFactor };
                cv::line( frame, from, to, colorSkeleton );
            }
        }
    };

    auto fs = CreateFrameStreamer<VideoStreamer>( videoPath.string( ), 100 );

    if ( fs )
        fs->Run( RunPoseEstimation );
}