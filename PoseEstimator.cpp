#include "PoseEstimator.hpp"

#include <chrono>

using namespace Logger;

namespace {

bool InitializeCudaBackend( Ort::SessionOptions& sessionOptions )
{
    auto& ortApi = Ort::GetApi( );
    OrtCUDAProviderOptionsV2* pCudaOptions = nullptr;
    ortApi.CreateCUDAProviderOptions( &pCudaOptions );
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype( ortApi.ReleaseCUDAProviderOptions )> cudaOptions(
        pCudaOptions, ortApi.ReleaseCUDAProviderOptions
    );
    std::vector<const char*> keys{ "device_id", "cudnn_conv_use_max_workspace", "do_copy_in_default_stream" };
    std::vector<const char*> values{ "0", "0", "1" };
    ortApi.UpdateCUDAProviderOptions( cudaOptions.get( ), keys.data( ), values.data( ), keys.size( ) );
    return nullptr == ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2( sessionOptions, cudaOptions.get( ) );
}

bool InitializeTensorRTBackend( Ort::SessionOptions& sessionOptions, const std::string& engineCachePath )
{
    auto& ortApi = Ort::GetApi( );
    OrtTensorRTProviderOptionsV2* pTrtOptions = nullptr;
    ortApi.CreateTensorRTProviderOptions( &pTrtOptions );
    std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype( ortApi.ReleaseTensorRTProviderOptions )> trtOptions(
        pTrtOptions, ortApi.ReleaseTensorRTProviderOptions
    );
    std::vector<const char*> trtKeys{
        "device_id",
        "trt_fp16_enable",
        "trt_dla_enable",
        "trt_dla_core",
        "trt_engine_cache_enable",
        "trt_engine_cache_path" };
    std::vector<const char*> trtValues{ "0", "1", "0", "1", "1", engineCachePath.c_str( ) };
    ortApi.UpdateTensorRTProviderOptions( trtOptions.get( ), trtKeys.data( ), trtValues.data( ), trtKeys.size( ) );
    return nullptr == ortApi.SessionOptionsAppendExecutionProvider_TensorRT_V2( sessionOptions, trtOptions.get( ) );
}

} // namespace

bool PoseEstimator::Initialize(
    const wchar_t* const modelFilePath, RuntimeBackend backend, const std::string& instanceName
)
{
    mEnv = Ort::Env( OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str( ) );

    Ort::SessionOptions sessionOptions;
    switch ( backend ) {
    case RuntimeBackend::Cuda:
        if ( !InitializeCudaBackend( sessionOptions ) ) {
            mLogger->Log( Priority::Error, "Cuda backend could not be initialized" );
            mInitializedModel = false;
            return false;
        }
        mLogger->Log( Priority::Info, "Cuda backend initialized" );
        break;
    case RuntimeBackend::TensorRT:
        if ( !InitializeTensorRTBackend( sessionOptions, "C:\\tmp\\" ) ) {
            mLogger->Log( Priority::Error, "TensorRT backend could not be initialized" );
            mInitializedModel = false;
            return false;
        }
        mLogger->Log( Priority::Info, "TensorRT backend initialized" );
        break;
    }

    try {
        mSession = Ort::Session( mEnv, modelFilePath, sessionOptions );
        mInitializedModel = true;
        LoadModelParameters( );
        if ( !DryRun( ) ) {
            mLogger->Log( Priority::Error, "DryRun did not complete successfully" );
            mInitializedModel = false;
            return mInitializedModel;
        }
    }
    catch ( const std::exception& e ) {
        mLogger->Log( Priority::Error, e.what( ) );
        mInitializedModel = false;
        return mInitializedModel;
    }

    mLogger->Log( Priority::Info, "Initialized successfully" );
    mInitializedModel = true;
    return mInitializedModel;
}

bool PoseEstimator::Forward(
    std::vector<Detection>& detections, float* frameData, int frameWidth, int frameHeight, int frameChannels
)
{
    if ( !mInitializedModel ) {
        mLogger->Log( Priority::Warning, "Running forward propagation on an uninitialized model" );
        return false;
    }

    auto inputSize = GetModelInputSize( );
    if ( ( frameData == nullptr ) || ( frameWidth != inputSize.width ) || ( frameHeight != inputSize.height )
         || ( frameChannels != inputSize.channels ) ) {
        mLogger->Log( Priority::Error, "Invalid input frame" );
        return false;
    }

    std::vector<float> tempIn( frameData, frameData + ( frameWidth * frameHeight * frameChannels ) );
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        frameData,
        frameWidth * frameHeight * frameChannels,
        mMp.inputTensorShape.data( ),
        mMp.inputTensorShape.size( )
    );
    try {
        std::vector<Ort::Value> outputTensors = mSession.Run(
            Ort::RunOptions{ nullptr },
            mMp.inputNodeNames.data( ),
            &inputTensor,
            mMp.numInputNodes,
            mMp.outputNodeNames.data( ),
            mMp.numOutputNodes
        );
        auto typeAndShapeInfo = outputTensors.front( ).GetTensorTypeAndShapeInfo( );
        std::vector<int64_t> shape = typeAndShapeInfo.GetShape( );
        const float* outputData = outputTensors.front( ).GetTensorData<float>( );

        if ( outputData != nullptr ) {
            detections.resize( shape.front( ) );
            memcpy(
                detections.data( ),
                outputData,
                sizeof( float ) * std::accumulate( shape.begin( ), shape.end( ), 1, std::multiplies<float>( ) )
            );
        }
    }
    catch ( const std::exception& e ) {
        mLogger->Log( Priority::Error, e.what( ) );
        return false;
    }
    return true;
}

float PoseEstimator::Benchmark( int numberOfIterations )
{
    if ( !mInitializedModel ) {
        mLogger->Log( Priority::Warning, "Benchmarking on an uninitialized model" );
        return -1.f;
    }

    const auto inputSize = GetModelInputSize( );
    std::unique_ptr<float[]> dummyImage =
        std::make_unique<float[]>( inputSize.width * inputSize.height * inputSize.channels );
    std::vector<PoseEstimator::Detection> detections;

    const auto start = std::chrono::high_resolution_clock::now( );
    for ( int i = 0; i < numberOfIterations; ++i ) {
        Forward( detections, dummyImage.get( ), inputSize.width, inputSize.height, inputSize.channels );
    }
    const auto end = std::chrono::high_resolution_clock::now( );
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count( );
    return elapsed / numberOfIterations;
}

PoseEstimator::InputSize PoseEstimator::GetModelInputSize( ) const
{
    InputSize size;
    size.channels = static_cast<int>( mMp.inputTensorShape[ 1 ] );
    size.width = static_cast<int>( mMp.inputTensorShape[ 2 ] );
    size.height = static_cast<int>( mMp.inputTensorShape[ 3 ] );
    return size;
}

// ##########################################################################################################

bool PoseEstimator::DryRun( )
{
    const auto inputSize = GetModelInputSize( );
    std::unique_ptr<float[]> dummyImage =
        std::make_unique<float[]>( inputSize.width * inputSize.height * inputSize.channels );
    std::vector<Detection> dummyOutput;
    return Forward( dummyOutput, dummyImage.get( ), inputSize.width, inputSize.height, inputSize.channels );
}

void PoseEstimator::LoadModelParameters( )
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