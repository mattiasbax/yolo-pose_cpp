#include "DrawUtils.hpp"

#include <opencv2/imgproc.hpp>

namespace DrawUtils {

cv::Mat DrawPosesInFrame(
    const cv::Size& frameSize,
    int frameType,
    const std::vector<PoseEstimator::Detection>& detections,
    const ScaleFactor& scaleFactor
)
{
    cv::Mat frame = cv::Mat::zeros( frameSize, frameType );
    if ( detections.empty( ) )
        return frame;

    const double confidenceThreshold = 0.3;

    for ( const auto& detection : detections ) {
        const cv::Scalar colorBox = { 200, 0, 0 };      // blue
        const cv::Scalar colorSkeleton = { 0, 200, 0 }; // green
        const cv::Scalar colorJoints = { 0, 0, 200 };   // red
        if ( detection.box.score < confidenceThreshold )
            continue;
        const cv::Point2f tl = { detection.box.tlX * scaleFactor.wFactor, detection.box.tlY * scaleFactor.hFactor };
        const cv::Point2f br = { detection.box.brX * scaleFactor.wFactor, detection.box.brY * scaleFactor.hFactor };
        cv::rectangle( frame, tl, br, colorBox, 2 );

        for ( int i = 0; i < detection.keyPoints.size( ); ++i ) {
            const auto& keypoint = detection.keyPoints[ i ];
            if ( keypoint.score < confidenceThreshold )
                continue;

            const cv::Point2f center = { keypoint.x * scaleFactor.wFactor, keypoint.y * scaleFactor.hFactor };
            cv::circle( frame, center, 3, colorJoints, -1 );
        }

        for ( const auto& edge : PoseEstimator::skeleton ) {
            if ( detection.keyPoints[ edge.first ].score < confidenceThreshold
                 || detection.keyPoints[ edge.second ].score < confidenceThreshold )
                continue;
            cv::Point2f from = {
                detection.keyPoints[ edge.first ].x * scaleFactor.wFactor,
                detection.keyPoints[ edge.first ].y * scaleFactor.hFactor };
            cv::Point2f to = {
                detection.keyPoints[ edge.second ].x * scaleFactor.wFactor,
                detection.keyPoints[ edge.second ].y * scaleFactor.hFactor };
            cv::line( frame, from, to, colorSkeleton );
        }
    }

    return frame;
}

} // namespace DrawUtils