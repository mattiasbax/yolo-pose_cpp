#include "FrameStreamer.hpp"

#include <chrono>
#include <future>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void FrameStreamer::Run( std::function<Result( const cv::Mat& inputFrame )> processFrame )
{
    using namespace std::chrono_literals;

    cv::Mat frame;
    int msWaitTime = 0;
    int processFrameTime = 0;

    const long long waitTime = 1 / ( mFps / 1000000.0 );
    const auto runStart = std::chrono::high_resolution_clock::now( );

    int count = 0;
    int keyPressed = 0;

    std::future<Result> modelOutput;
    Result mostRecentResult;
    while ( AcquireFrame( frame ) && !( keyPressed == 'q' || keyPressed == 'Q' ) ) {
        auto nextTick = runStart + ( ++count * std::chrono::microseconds( waitTime ) );

        if ( processFrame ) {
            if ( !modelOutput.valid( ) ) {
                // TODO: std::async introduces too much overhead. Switch to a pure jthread solution
                modelOutput = std::async( std::launch::async, processFrame, frame );
            }
        }

        std::this_thread::sleep_until( nextTick );

        if ( processFrame ) {
            if ( modelOutput.wait_for( 0s ) == std::future_status::ready ) {
                mostRecentResult = modelOutput.get( );
            }
            DrawUtils::DrawPosesInFrame( frame, mostRecentResult.modelOutput, mostRecentResult.scaleFactor );
        }

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