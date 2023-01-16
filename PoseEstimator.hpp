#pragma once

#include <array>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <vector>


class YoloV7Pose
{
  public:
    struct InputSize {
        int width;
        int height;
        int channels;
    };

    struct BoundingBox {
        float tlX;
        float tlY;
        float brX;
        float brY;
        float score;
        float label;
    };

    struct KeyPoint {
        float x;
        float y;
        float score;
    };

    enum Joint {
        Nose = 0,
        leftEye = 1,
        rightEye = 2,
        leftEar = 3,
        rightEar = 4,
        leftShoulder = 5,
        rightShoulder = 6,
        leftElbow = 7,
        rightElbow = 8,
        leftWrist = 9,
        rightWrist = 10,
        leftHip = 11,
        rightHip = 12,
        leftKnee = 13,
        rightKnee = 14,
        leftAnkle = 15,
        rightAnkle = 16,
    };

    struct Detection {
        BoundingBox box;
        std::array<KeyPoint, 17> keyPoints;
    };

    YoloV7Pose( ) : mEnv( nullptr ), mSession( nullptr ), mInitializedModel( false ) {}

    virtual bool Initialize( const wchar_t* const modelFilePath, const std::string& instanceName = "Model" )
    {
        mEnv = Ort::Env( OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str( ) );
        Ort::SessionOptions sessionOptions;
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA( cudaOptions );
        try {
            mSession = Ort::Session( mEnv, modelFilePath, sessionOptions );
            mInitializedModel = true;
            LoadModelParameters( );
            if ( !DryRun( ) ) {
                std::cout << "Error: DryRun did not complete successfully" << std::endl;
                mInitializedModel = false;
            }
        } catch ( const std::exception& e ) {
            std::cout << "Error: " << e.what( ) << std::endl;
            mInitializedModel = false;
        }
        return mInitializedModel;
    }

    bool Forward( std::vector<Detection>& detections, float* frameData, int frameWidth, int frameHeight,
                  int frameChannels )
    {
        if ( !mInitializedModel )
            return false;

        auto inputSize = GetModelInputSize( );
        if ( ( frameData == nullptr ) || ( frameWidth != inputSize.width ) || ( frameHeight != inputSize.height )
             || ( frameChannels != inputSize.channels ) )
            return false;

        std::vector<float> tempIn( frameData, frameData + ( frameWidth * frameHeight * frameChannels ) );
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>( memoryInfo,
                                                                        frameData,
                                                                        frameWidth * frameHeight * frameChannels,
                                                                        mMp.inputTensorShape.data( ),
                                                                        mMp.inputTensorShape.size( ) );
        try {
            std::vector<Ort::Value> outputTensors = mSession.Run( Ort::RunOptions{ nullptr },
                                                                  mMp.inputNodeNames.data( ),
                                                                  &inputTensor,
                                                                  mMp.numInputNodes,
                                                                  mMp.outputNodeNames.data( ),
                                                                  mMp.numOutputNodes );
            auto typeAndShapeInfo = outputTensors.front( ).GetTensorTypeAndShapeInfo( );
            std::vector<int64_t> shape = typeAndShapeInfo.GetShape( );
            const float* outputData = outputTensors.front( ).GetTensorData<float>( );

            if ( outputData != nullptr ) {
                detections.resize( shape.front( ) );
                memcpy( detections.data( ),
                        outputData,
                        sizeof( float )
                            * std::accumulate( shape.begin( ), shape.end( ), 1, std::multiplies<float>( ) ) );
            }
        } catch ( const std::exception& ) {
            return false;
        }
        return true;
    }

    bool DryRun( )
    {
        auto inputSize = GetModelInputSize( );
        std::unique_ptr<float[]> dummyImage =
            std::make_unique<float[]>( inputSize.width * inputSize.height * inputSize.channels );
        std::vector<Detection> dummyOutput;
        return Forward( dummyOutput, dummyImage.get( ), inputSize.width, inputSize.height, inputSize.channels );
    }

    InputSize GetModelInputSize( ) const
    {
        InputSize size;
        size.channels = static_cast<int>( mMp.inputTensorShape[ 1 ] );
        size.width = static_cast<int>( mMp.inputTensorShape[ 2 ] );
        size.height = static_cast<int>( mMp.inputTensorShape[ 3 ] );
        return size;
    }

  private:
    Ort::Env mEnv;
    Ort::Session mSession;
    bool mInitializedModel;

    struct ModelParameters {
        size_t numInputNodes;
        size_t numOutputNodes;
        std::vector<Ort::AllocatedStringPtr> inputNodeNamesAllocated;
        std::vector<const char*> inputNodeNames;
        std::vector<Ort::AllocatedStringPtr> outputNodeNamesAllocated;
        std::vector<const char*> outputNodeNames;
        std::vector<int64_t> inputTensorShape;
    };

    void LoadModelParameters( )
    {
        Ort::AllocatorWithDefaultOptions allocator;
        mMp.numInputNodes = mSession.GetInputCount( );
        for ( size_t idx = 0; idx < mMp.numInputNodes; ++idx ) {
            mMp.inputNodeNamesAllocated.push_back( mSession.GetInputNameAllocated( idx, allocator ) );
            mMp.inputNodeNames.push_back( mMp.inputNodeNamesAllocated.back( ).get( ) );
        }

        mMp.numOutputNodes = mSession.GetOutputCount( );
        for ( size_t idx = 0; idx < mMp.numOutputNodes; ++idx ) {
            mMp.outputNodeNamesAllocated.push_back( mSession.GetOutputNameAllocated( idx, allocator ) );
            mMp.outputNodeNames.push_back( mMp.outputNodeNamesAllocated.back( ).get( ) );
        }

        mMp.inputTensorShape = mSession.GetInputTypeInfo( 0 ).GetTensorTypeAndShapeInfo( ).GetShape( );
    }

    ModelParameters mMp;
};
