#include "FrameStreamer.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>

void FrameStreamer::Run( std::function<void( const cv::Mat& inputFrame )> processFrame )
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
    } while ( FrameStreamer::VisualizeStream(
        frame, ( 1000 / mFps ) - msWaitTime ) ); // TODO: Make a thread running at specified FPS
}

bool FrameStreamer::VisualizeStream( const cv::Mat& frame, int msWaitTime ) // TODO: Move to ano namespace
{
    if ( frame.empty( ) )
        return false;
    cv::imshow( mWindowName, frame );
    const int keyPressed = cv::waitKey( 1000 / mFps );
    if ( keyPressed == 'q' || keyPressed == 'Q' )
        return false;
    return true;
}

// ##################################

bool ImageStreamer::Initialize( )
{
    try {
        mImage = cv::imread( mImageFilePath );
        mIsInitialized = true;
    } catch ( const std::exception& ) {
        mIsInitialized = false;
    }
    return mIsInitialized;
}

bool ImageStreamer::AcquireFrame( cv::Mat& frame )
{
    if ( !mIsInitialized )
        return false;
    frame = mImage.clone( );
    return true;
}

// ##################################

bool VideoStreamer::Initialize( )
{
    try {
        mCap = cv::VideoCapture( mVideoFilePath );
        mIsInitialized = mCap.isOpened( );
    } catch ( const std::exception& ) {
        mIsInitialized = false;
    }
    return mIsInitialized;
}

bool VideoStreamer::AcquireFrame( cv::Mat& frame )
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