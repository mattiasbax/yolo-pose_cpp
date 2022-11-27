#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory.h>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class ConvNetBase
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

    ConvNetBase( ) : mEnv( nullptr ), mSession( nullptr ), mInitializedModel( false ) {}

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

    bool Forward( std::vector<Detection>& detections, float* frameData, int frameWidth, int frameHeight, int frameChannels )
    {
        if ( !mInitializedModel )
            return false;

        auto inputSize = GetModelInputSize( );
        if ( ( frameData == nullptr ) || ( frameWidth != inputSize.width ) || ( frameHeight != inputSize.height ) || ( frameChannels != inputSize.channels ) )
            return false;

        std::vector<float> tempIn( frameData, frameData + ( frameWidth * frameHeight * frameChannels ) );
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>( memoryInfo, frameData, frameWidth * frameHeight * frameChannels, mMp.inputTensorShape.data( ), mMp.inputTensorShape.size( ) );
        try {
            std::vector<Ort::Value> outputTensors =
                mSession.Run( Ort::RunOptions{ nullptr }, mMp.inputNodeNames.data( ), &inputTensor, mMp.numInputNodes, mMp.outputNodeNames.data( ), mMp.numOutputNodes );
            auto typeAndShapeInfo = outputTensors.front( ).GetTensorTypeAndShapeInfo( );
            std::vector<int64_t> shape = typeAndShapeInfo.GetShape( );
            const float* outputData = outputTensors.front( ).GetTensorData<float>( );

            if ( outputData != nullptr ) {
                detections.resize( shape.front( ) );
                memcpy( detections.data( ), outputData, sizeof( float ) * std::accumulate( shape.begin( ), shape.end( ), 1, std::multiplies<float>( ) ) );
            }
        } catch ( const std::exception& ) {
            return false;
        }
        return true;
    }

    bool DryRun( )
    {
        auto inputSize = GetModelInputSize( );
        std::unique_ptr<float[]> dummyImage = std::make_unique<float[]>( inputSize.width * inputSize.height * inputSize.channels );
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

class YoloV7Pose final : public ConvNetBase
{
  public:
    YoloV7Pose( ) = default;
    ~YoloV7Pose( ) = default;

  private:
};

int main( )
{
    std::filesystem::path modelPath = __FILE__;
    modelPath.remove_filename( ).append( "Yolov5s6_pose_640.onnx" );

    YoloV7Pose model;
    model.Initialize( modelPath.wstring( ).c_str( ), "yolo-pose" );

    std::filesystem::path imgPath = __FILE__;
    imgPath.remove_filename( ).append( "img.png" );
    cv::Mat inputImage = cv::imread( imgPath.string( ) );
    cv::resize( inputImage, inputImage, cv::Size( 640, 640 ) ); // resize to network image size

    int size[] = { 1, 3, 640, 640 };
    cv::Mat inputBlob( 4, size, CV_32F );
    cv::dnn::blobFromImage( inputImage, inputBlob, 0.00392156862745098, cv::Size( 640, 640 ), cv::Scalar( 0, 0, 0, 0 ), true, false, CV_32F );

    std::vector<ConvNetBase::Detection> output;
    constexpr int numIterations = 100;
    const auto before = std::chrono::steady_clock::now( );
    for ( int i = 0; i < numIterations; ++i ) {
        model.Forward( output, (float*) inputBlob.data, 640, 640, 3 );
    }
    const auto after = std::chrono::steady_clock::now( );
    std::cout << "FPS: ";
    std::cout << 1000 / ( std::chrono::duration_cast<std::chrono::milliseconds>( after - before ).count( ) / numIterations ) << std::endl;

    const std::vector<std::pair<ConvNetBase::Joint, ConvNetBase::Joint>> skeleton{ { ConvNetBase::Joint::leftAnkle, ConvNetBase::Joint::leftKnee },
                                                                                   { ConvNetBase::Joint::leftKnee, ConvNetBase::Joint::leftHip },
                                                                                   { ConvNetBase::Joint::rightAnkle, ConvNetBase::Joint::rightKnee },
                                                                                   { ConvNetBase::Joint::rightKnee, ConvNetBase::Joint::rightHip },
                                                                                   { ConvNetBase::Joint::leftHip, ConvNetBase::Joint::rightHip },
                                                                                   { ConvNetBase::Joint::leftShoulder, ConvNetBase::Joint::leftHip },
                                                                                   { ConvNetBase::Joint::rightShoulder, ConvNetBase::Joint::rightHip },
                                                                                   { ConvNetBase::Joint::leftShoulder, ConvNetBase::Joint::rightShoulder },
                                                                                   { ConvNetBase::Joint::leftShoulder, ConvNetBase::Joint::leftElbow },
                                                                                   { ConvNetBase::Joint::rightShoulder, ConvNetBase::Joint::rightElbow },
                                                                                   { ConvNetBase::Joint::leftElbow, ConvNetBase::Joint::leftWrist },
                                                                                   { ConvNetBase::Joint::rightElbow, ConvNetBase::Joint::rightWrist },
                                                                                   { ConvNetBase::Joint::leftEye, ConvNetBase::Joint::rightEye },
                                                                                   { ConvNetBase::Joint::Nose, ConvNetBase::Joint::leftEye },
                                                                                   { ConvNetBase::Joint::Nose, ConvNetBase::Joint::rightEye },
                                                                                   { ConvNetBase::Joint::leftEye, ConvNetBase::Joint::leftEar },
                                                                                   { ConvNetBase::Joint::rightEye, ConvNetBase::Joint::rightEar },
                                                                                   { ConvNetBase::Joint::leftEar, ConvNetBase::Joint::leftShoulder },
                                                                                   { ConvNetBase::Joint::rightEar, ConvNetBase::Joint::rightShoulder } };

    for ( const auto& detection : output ) {
        const cv::Scalar colorBox = { 200, 0, 0 };      // blue
        const cv::Scalar colorSkeleton = { 0, 200, 0 }; // green
        const cv::Scalar colorJoints = { 0, 0, 200 };   // red
        if ( detection.box.score < 0.5 )
            continue;
        const cv::Point2f tl = { detection.box.tlX, detection.box.tlY };
        const cv::Point2f br = { detection.box.brX, detection.box.brY };
        cv::rectangle( inputImage, tl, br, colorBox, 2 );

        for ( int i = 0; i < detection.keyPoints.size( ); ++i ) {
            const auto& keypoint = detection.keyPoints[ i ];
            if ( keypoint.score < 0.5 )
                continue;

            const cv::Point2f center = { keypoint.x, keypoint.y };
            cv::circle( inputImage, center, 3, colorJoints, -1 );
        }

        for ( const auto& edge : skeleton ) {
            if ( detection.keyPoints[ edge.first ].score < 0.5 || detection.keyPoints[ edge.second ].score < 0.5 )
                continue;
            cv::Point2f from = { detection.keyPoints[ edge.first ].x, detection.keyPoints[ edge.first ].y };
            cv::Point2f to = { detection.keyPoints[ edge.second ].x, detection.keyPoints[ edge.second ].y };
            cv::line( inputImage, from, to, colorSkeleton );
        }
    }

    cv::imshow( "result", inputImage );
    cv::waitKey( 0 );
}