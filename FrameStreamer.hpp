#pragma once

#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

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
    FrameStreamer( ) : mFps( 0. ) { }

    virtual ~FrameStreamer( ) = default;

    virtual bool Initialize( ) = 0;

    virtual bool AcquireFrame( cv::Mat& frame ) = 0;

    void Run( std::function<void( const cv::Mat& inputFrame )> processFrame = nullptr );

private:
    bool VisualizeStream( const cv::Mat& frame, int msWaitTime );
    static constexpr std::string mWindowName = "Stream";

protected:
    float mFps;
};

// ##################################

class ImageStreamer final : public FrameStreamer {
public:
    ImageStreamer( const std::string& imageFilePath ) : mIsInitialized( false ), mImageFilePath( imageFilePath ) { }

    bool Initialize( ) override;

    bool AcquireFrame( cv::Mat& frame ) override;

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

    bool AcquireFrame( cv::Mat& frame ) override;

private:
    bool mIsInitialized;
    const std::string mVideoFilePath;
    const bool mLoopVideo;
    cv::VideoCapture mCap;
};
