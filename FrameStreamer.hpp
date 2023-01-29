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
concept IsStreamer = IsAnyOf<T, ImageStreamer, VideoStreamer>;

template <typename T>
    requires IsStreamer<T>
std::unique_ptr<FrameStreamer> CreateFrameStreamer( const std::string fileName, int frameRate )
{
    auto streamer = std::make_unique<T>( fileName, frameRate );

    if ( streamer->Initialize( ) )
        return streamer;
    else
        return nullptr;
}

// ##################################

class FrameStreamer
{
  public:
    FrameStreamer( int fps ) : mFps( fps ), mSuppressWarnings( true ) {}

    virtual ~FrameStreamer( ) = default;

    virtual bool Initialize( ) = 0;

    virtual bool AcquireFrame( cv::Mat& frame ) = 0;

    void ToggleWarnings( ) { mSuppressWarnings = !mSuppressWarnings; };

    void Run( std::function<void( const cv::Mat& inputFrame )> processFrame = nullptr );

  private:
    bool VisualizeStream( const cv::Mat& frame, int msWaitTime );

    static constexpr std::string mWindowName = "Stream";
    const int mFps;
    bool mSuppressWarnings;
};

// ##################################

class ImageStreamer final : public FrameStreamer
{
  public:
    ImageStreamer( const std::string& imageFilePath, int fps = 30 )
        : FrameStreamer( fps ), mIsInitialized( false ), mImageFilePath( imageFilePath )
    {
    }

    bool Initialize( ) override;

    bool AcquireFrame( cv::Mat& frame ) override;

  private:
    bool mIsInitialized;
    const std::string mImageFilePath;
    cv::Mat mImage;
};

// ##################################

class VideoStreamer final : public FrameStreamer
{
  public:
    VideoStreamer( const std::string& videoFilePath, int fps = 30, bool loopVideo = true )
        : FrameStreamer( fps ), mIsInitialized( false ), mLoopVideo( loopVideo ), mVideoFilePath( videoFilePath )
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

// TODO: Implement a CameraStreamer
// TODO: Split into declaration/definition
// TODO: Add a frame counter in the image
// TODO: Pimpl to avoid exposing cv::videoio outwards
// TODO: Capture if trying to load image to video streamer
// TODO: Returns silently if cannot find video file
// TODO: Resize video to specified size