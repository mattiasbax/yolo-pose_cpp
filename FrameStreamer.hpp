#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <string>

class FrameStreamer
{
  public:
    FrameStreamer( int fps ) : mFps( fps ), mSuppressWarnings( true ) {}

    virtual ~FrameStreamer( ) = default;

    virtual bool Initialize( ) = 0;

    virtual bool AcquireFrame( cv::Mat& frame ) = 0;

    void ToggleWarnings( ) { mSuppressWarnings = !mSuppressWarnings; };

    void Run( std::function<void( const cv::Mat& inputFrame )> processFrame = nullptr )
    {
        cv::Mat frame;
        int msWaitTime = 0;
        int processFrameTime = 0;
        do {
            if ( !AcquireFrame( frame ) )
                break;

            const auto start = std::chrono::high_resolution_clock::now( );
            if ( processFrame ) {
                processFrame( frame );
            }
            const auto end = std::chrono::high_resolution_clock::now( );
            int processFrameTime =
                static_cast<int>( std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count( ) );

            msWaitTime = std::max<int>( 0, ( 1000 / mFps ) - processFrameTime );
            if ( !mSuppressWarnings && msWaitTime == 0 ) {
                std::cout << "Warning: Processed frame took " << processFrameTime << " ms." << std::endl;
            }
        } while ( FrameStreamer::VisualizeStream( frame, ( 1000 / mFps ) - msWaitTime ) );
    }

  private:
    bool VisualizeStream( const cv::Mat& frame, int msWaitTime ) // TODO: Move to ano namespace
    {
        if ( frame.empty( ) )
            return false;
        cv::imshow( mWindowName, frame );
        const int keyPressed = cv::waitKey( 1000 / mFps );
        if ( keyPressed == 'q' || keyPressed == 'Q' )
            return false;
        return true;
    }

    static constexpr std::string mWindowName = "Stream";
    const int mFps;
    bool mSuppressWarnings;
};

class ImageStreamer : public FrameStreamer
{
  public:
    ImageStreamer( const std::string& imageFilePath, int fps = 30 )
        : FrameStreamer( fps ), mIsInitialized( false ), mImageFilePath( imageFilePath )
    {
    }

    bool Initialize( ) override
    {
        try {
            mImage = cv::imread( mImageFilePath );
            mIsInitialized = true;
        } catch ( const std::exception& ) {
            mIsInitialized = false;
        }
        return mIsInitialized;
    }

    bool AcquireFrame( cv::Mat& frame ) override
    {
        if ( !mIsInitialized )
            return false;
        frame = mImage.clone( );
        return true;
    }

  private:
    bool mIsInitialized;
    const std::string mImageFilePath;
    cv::Mat mImage;
};

class VideoStreamer : public FrameStreamer
{
  public:
    VideoStreamer( const std::string& videoFilePath, int fps = 30, bool loopVideo = true )
        : FrameStreamer( fps ), mIsInitialized( false ), mLoopVideo( loopVideo ), mVideoFilePath( videoFilePath )
    {
    }

    ~VideoStreamer( )
    {
        if ( mIsInitialized ) {
            mCap.release( );
        }
    }

    bool Initialize( ) override
    {
        try {
            mCap = cv::VideoCapture( mVideoFilePath );
            mIsInitialized = mCap.isOpened( );
        } catch ( const std::exception& ) {
            mIsInitialized = false;
        }
        return mIsInitialized;
    }

    bool AcquireFrame( cv::Mat& frame ) override
    {
        if ( !mIsInitialized )
            return false;

        mCap >> frame;
        if ( frame.empty( ) ) {
            if ( mLoopVideo ) {
                mCap.set( cv::CAP_PROP_POS_FRAMES, 0 );
                mCap >> frame;
            } else
                return false;
        }
        return true;
    }

  private:
    bool mIsInitialized;
    const std::string mVideoFilePath;
    const bool mLoopVideo;
    cv::VideoCapture mCap;
};

// TODO: Implement a CameraStreamer
// TODO: Implement a FrameStreamerFactory
// TODO: Split into declaration/definition
// TODO: Add a frame counter in the image