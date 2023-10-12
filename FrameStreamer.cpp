#include "FrameStreamer.hpp"

#include <chrono>
#include <future>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

void ProcessFrame(
    cv::Mat& frame, FrameStreamer::FrameProcessFunction f, std::future<FrameStreamer::Result>& processResult
)
{
    if ( f && !processResult.valid( ) ) {
        processResult = std::async(
            std::launch::async, f, frame
        ); // TODO: Need to ensure that frame is not overwritten before processed
    }
}

bool GetProcessedFrameResult(
    FrameStreamer::Result& result,
    FrameStreamer::FrameProcessFunction f,
    std::future<FrameStreamer::Result>& processResult
)
{
    using namespace std::chrono_literals;
    if ( f ) {
        if ( processResult.valid( ) && processResult.wait_for( 0ms ) == std::future_status::ready ) {
            result = processResult.get( );
            return true;
        }
    }
    return false;
}

} // namespace

void FrameStreamer::Run( FrameProcessFunction f )
{
    enum class State {
        Running,
        Paused
    };

    using namespace std::chrono_literals;
    using enum State;

    cv::Mat frame;
    cv::Mat poseFrame;
    int frameNumber = -1;

    int count = 0;
    int keyPressed = 0;

    std::future<Result> processResult;
    Result mostRecentResult;

    State s = Running;
    const long long waitTime = 1 / ( mFps / 1000000.0 );
    const auto runStart = std::chrono::high_resolution_clock::now( );

    while ( !( keyPressed == 'q' || keyPressed == 'Q' ) ) {
        auto nextTick = runStart + ( ++count * std::chrono::microseconds( waitTime ) );
        switch ( s ) {
        case Running:
            if ( keyPressed == 'p' || keyPressed == 'P' ) {
                s = Paused;
                break;
            }
            frameNumber = AcquireNextFrame( frame );
            if ( frameNumber < 0 )
                return;
            ProcessFrame( frame, f, processResult );
            break;
        case Paused:
            if ( keyPressed == 'r' || keyPressed == 'R' ) {
                s = Running;
            }
            else if ( keyPressed == 'f' || keyPressed == 'F' ) {
                frameNumber = AcquireNextFrame( frame );
                if ( frameNumber < 0 )
                    return;
                ProcessFrame( frame, f, processResult );
            }
            else if ( keyPressed == 'b' || keyPressed == 'B' ) {
                frameNumber = AcquirePreviousFrame( frame );
                if ( frameNumber < 0 )
                    return;
                ProcessFrame( frame, f, processResult );
            }
            break;
        }

        std::this_thread::sleep_until( nextTick );

        if ( GetProcessedFrameResult( mostRecentResult, f, processResult ) ) {
            poseFrame = DrawUtils::DrawPosesInFrame(
                frame.size( ), frame.type( ), mostRecentResult.modelOutput, mostRecentResult.scaleFactor
            );
        }

        // ! Draw framenumber in fram here
        if ( !poseFrame.empty( ) ) {
            cv::imshow( mWindowName, frame + poseFrame );
        }
        else {
            cv::imshow( mWindowName, frame );
        }
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
        mNumberOfFrames = 1;
    }
    catch ( const std::exception& ) {
        mIsInitialized = false;
    }
    return mIsInitialized;
}

int ImageStreamer::AcquireNextFrame( cv::Mat& frame )
{
    if ( !mIsInitialized )
        return -1;
    frame = mImage.clone( );
    return 1;
}

int ImageStreamer::AcquirePreviousFrame( cv::Mat& frame )
{
    // * There are no 'previous' frames in an image stream
    return AcquireNextFrame( frame );
}

// ##################################

bool VideoStreamer::Initialize( )
{
    try {
        mCap = cv::VideoCapture( mVideoFilePath );
        mIsInitialized = mCap.isOpened( );
        if ( mIsInitialized ) {
            mFps = mCap.get( cv::CAP_PROP_FPS );
            mNumberOfFrames = mCap.get( cv::CAP_PROP_FRAME_COUNT );
        }
    }
    catch ( const std::exception& ) {
        mIsInitialized = false;
    }
    return mIsInitialized;
}

int VideoStreamer::AcquireNextFrame( cv::Mat& frame )
{
    if ( !mIsInitialized )
        return false;

    const int currentFrame = mCap.get( cv::CAP_PROP_POS_FRAMES );
    if ( mLoopVideo ) {
        if ( currentFrame >= mNumberOfFrames ) {
            mCap.set( cv::CAP_PROP_POS_FRAMES, 0 );
        }
    }
    mCap >> frame;
    return frame.empty( ) ? -1 : currentFrame;
}

int VideoStreamer::AcquirePreviousFrame( cv::Mat& frame )
{
    if ( !mIsInitialized )
        return false;

    const int currentFrame = mCap.get( cv::CAP_PROP_POS_FRAMES );
    if ( currentFrame != 0 ) {
        mCap.set( cv::CAP_PROP_POS_FRAMES, currentFrame - 2 );
    }
    else {
        mCap.set( cv::CAP_PROP_POS_FRAMES, mNumberOfFrames );
    }
    mCap >> frame;
    return frame.empty( ) ? -1 : currentFrame;
}