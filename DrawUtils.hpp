#pragma once

#include "PoseEstimator.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace DrawUtils {

struct ScaleFactor {
    float wFactor;
    float hFactor;
};

cv::Mat DrawPosesInFrame(
    const cv::Size& frameSize,
    int frameType,
    const std::vector<PoseEstimator::Detection>& detections,
    const ScaleFactor& scaleFactor = { .wFactor = 1.f, .hFactor = 1.f }
);

} // namespace DrawUtils