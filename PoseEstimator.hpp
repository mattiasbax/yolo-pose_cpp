#pragma once

#include <array>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <vector>

class PoseEstimator {
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

    struct Detection {
        BoundingBox box;
        std::array<KeyPoint, 17> keyPoints;
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

    using JointConnection = std::pair<Joint, Joint>;
    static constexpr std::array<JointConnection, 19> skeleton{
        JointConnection{ Joint::leftAnkle, Joint::leftKnee },
        JointConnection{ Joint::leftKnee, Joint::leftHip },
        JointConnection{ Joint::rightAnkle, Joint::rightKnee },
        JointConnection{ Joint::rightKnee, Joint::rightHip },
        JointConnection{ Joint::leftHip, Joint::rightHip },
        JointConnection{ Joint::leftShoulder, Joint::leftHip },
        JointConnection{ Joint::rightShoulder, Joint::rightHip },
        JointConnection{ Joint::leftShoulder, Joint::rightShoulder },
        JointConnection{ Joint::leftShoulder, Joint::leftElbow },
        JointConnection{ Joint::rightShoulder, Joint::rightElbow },
        JointConnection{ Joint::leftElbow, Joint::leftWrist },
        JointConnection{ Joint::rightElbow, Joint::rightWrist },
        JointConnection{ Joint::leftEye, Joint::rightEye },
        JointConnection{ Joint::Nose, Joint::leftEye },
        JointConnection{ Joint::Nose, Joint::rightEye },
        JointConnection{ Joint::leftEye, Joint::leftEar },
        JointConnection{ Joint::rightEye, Joint::rightEar },
        JointConnection{ Joint::leftEar, Joint::leftShoulder },
        JointConnection{ Joint::rightEar, Joint::rightShoulder } };

    PoseEstimator( ) : mEnv( nullptr ), mSession( nullptr ), mInitializedModel( false ) { }

    virtual ~PoseEstimator( ) = default;

    bool Initialize( const wchar_t* const modelFilePath, const std::string& instanceName = "Model" );

    bool Forward(
        std::vector<Detection>& detections, float* frameData, int frameWidth, int frameHeight, int frameChannels
    );

    bool DryRun( );

    InputSize GetModelInputSize( ) const;

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

    void LoadModelParameters( );

    ModelParameters mMp;
};
