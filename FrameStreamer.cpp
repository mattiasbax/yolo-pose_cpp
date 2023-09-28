#include "FrameStreamer.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>

void FrameStreamer::Run( std::function<void( const cv::Mat& inputFrame )> processFrame )
{
    cv::Mat frame;
    int msWaitTime = 0;
    int processFrameTime = 0;

    const long long waitTime = 1 / ( mFps / 1000000.0 );
    const auto runStart = std::chrono::high_resolution_clock::now( );

    int count = 0;
    int keyPressed = 0;
    while ( AcquireFrame( frame ) && !( keyPressed == 'q' || keyPressed == 'Q' ) ) {
        // * DO WORK HERE *
        if ( processFrame ) {
            const auto processStart = std::chrono::high_resolution_clock::now( );
            processFrame( frame );
            const auto processEnd = std::chrono::high_resolution_clock::now( );
            int processDuration = duration_cast<std::chrono::microseconds>( processEnd - processStart ).count( );
        }

        std::this_thread::sleep_until( runStart + ( ++count * std::chrono::microseconds( waitTime ) ) );

        cv::imshow( mWindowName, frame );
        keyPressed = cv::waitKey( 1 );
    }
}

// ##################################

bool ImageStreamer::Initialize( )
{
    try {
        mImage = cv::imread( mImageFilePath );
        mIsInitialized = true;
        mFps = 30.f;
    }
    catch ( const std::exception& ) {
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
        if ( mIsInitialized ) {
            mFps = mCap.get( cv::CAP_PROP_FPS );
        }
    }
    catch ( const std::exception& ) {
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
        }
        else
            return false;
    }
    return true;
}