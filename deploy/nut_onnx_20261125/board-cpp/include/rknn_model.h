#pragma once
// RKNN 模型封装：加载 .rknn，喂入 RGB uint8 图像，输出 patches 描述子（float32）。
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

class RknnModel {
public:
    RknnModel() = default;
    ~RknnModel();

    bool load(const std::string& model_path);
    // rgb: BGR 或 RGB 均可，内部统一转 RGB 并缩放到模型输入尺寸；out 为 [N*C] float32。
    bool infer(const cv::Mat& img, std::vector<float>& out) const;

    int input_width() const { return in_w_; }
    int input_height() const { return in_h_; }

private:
    rknn_context ctx_ = 0;
    bool loaded_ = false;
    int in_w_ = 256;
    int in_h_ = 256;
    int out_elems_ = 0;
    std::vector<uint8_t> model_data_;
};
