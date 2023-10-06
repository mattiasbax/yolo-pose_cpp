#pragma once

#include "DrawUtils.hpp"
#include "PoseEstimator.hpp"

#include <functional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

// ##############################

class FrameStreamer;
class ImageStreamer;
class VideoStreamer;

template <typename T, typename... Ts>
concept IsAnyOf = ( std::same_as<T, Ts> || ... );

template <typename T>
concept Streamer = IsAnyOf<T, ImageStreamer, VideoStreamer>;

template <Streamer T>
std::unique_ptr<FrameStreamer> CreateFrameStreamer( const std::string fileName )
{
    auto streamer = std::make_unique<T>( fileName );

    if ( streamer->Initialize( ) )
        return streamer;
    else
        return nullptr;
}

// ##################################

class FrameStreamer {
public:
    FrameStreamer( ) : mFps( 0. ), mNumberOfFrames( 0 ) { }

    virtual ~FrameStreamer( ) = default;

    virtual bool Initialize( ) = 0;

    // TODO: Make the result type more generic and not pose estimation dependant
    struct Result {
        std::vector<PoseEstimator::Detection> modelOutput;
        DrawUtils::ScaleFactor scaleFactor;
    };

    using FrameProcessFunction = std::function<Result( const cv::Mat& inputFrame )>;
    void Run( FrameProcessFunction f = nullptr );

protected:
    float mFps;
    int mNumberOfFrames;

private:
    virtual bool AcquireNextFrame( cv::Mat& frame ) = 0;

    virtual bool AcquirePreviousFrame( cv::Mat& frame ) = 0;

    static constexpr std::string mWindowName = "Stream";
};

// ##################################

class ImageStreamer final : public FrameStreamer {
public:
    ImageStreamer( const std::string& imageFilePath ) : mIsInitialized( false ), mImageFilePath( imageFilePath ) { }

    bool Initialize( ) override;

    bool AcquireNextFrame( cv::Mat& frame ) override;

    bool AcquirePreviousFrame( cv::Mat& frame ) override;

private:
    bool mIsInitialized;
    const std::string mImageFilePath;
    cv::Mat mImage;
};

// ##################################

class VideoStreamer final : public FrameStreamer {
public:
    VideoStreamer( const std::string& videoFilePath ) :
        mIsInitialized( false ),
        mLoopVideo( true ),
        mVideoFilePath( videoFilePath )
    {
    }

    bool Initialize( ) override;

    bool AcquireNextFrame( cv::Mat& frame ) override;

    bool AcquirePreviousFrame( cv::Mat& frame ) override;

private:
    bool mIsInitialized;
    const std::string mVideoFilePath;
    const bool mLoopVideo;
    cv::VideoCapture mCap;
};
