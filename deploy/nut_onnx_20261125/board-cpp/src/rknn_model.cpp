#include "rknn_model.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

RknnModel::~RknnModel() {
    if (loaded_) {
        rknn_destroy(ctx_);
        loaded_ = false;
    }
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
}

bool RknnModel::load(const std::string& model_path) {
    model_data_ = read_file(model_path);
    if (model_data_.empty()) {
        std::cerr << "[RKNN] 模型文件读取失败: " << model_path << "\n";
        return false;
    }

    int ret = rknn_init(&ctx_, model_data_.data(),
                        static_cast<uint32_t>(model_data_.size()), 0, nullptr);
    if (ret < 0) {
        std::cerr << "[RKNN] rknn_init 失败: " << ret << "\n";
        return false;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cerr << "[RKNN] 查询 IO 数量失败: " << ret << "\n";
        return false;
    }

    rknn_tensor_attr in_attr;
    std::memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    in_w_ = in_attr.dims[2];
    in_h_ = in_attr.dims[1];

    rknn_tensor_attr out_attr;
    std::memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    out_elems_ = static_cast<int>(out_attr.n_elems);

    loaded_ = true;
    std::cout << "[RKNN] 加载成功 输入 " << in_w_ << "x" << in_h_
              << " 输出元素数 " << out_elems_ << "\n";
    return true;
}

bool RknnModel::infer(const cv::Mat& img, std::vector<float>& out) const {
    if (!loaded_) return false;

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    if (rgb.cols != in_w_ || rgb.rows != in_h_) {
        cv::resize(rgb, rgb, cv::Size(in_w_, in_h_));
    }
    if (!rgb.isContinuous()) rgb = rgb.clone();

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = rgb.data;
    inputs[0].size = static_cast<uint32_t>(rgb.total() * rgb.channels());
    inputs[0].pass_through = 0;                 // 由 NPU 完成量化/归一化
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0) {
        std::cerr << "[RKNN] inputs_set 失败: " << ret << "\n";
        return false;
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[RKNN] rknn_run 失败: " << ret << "\n";
        return false;
    }

    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;                  // 反量化为 float32
    ret = rknn_outputs_get(ctx_, 1, outputs, nullptr);
    if (ret < 0) {
        std::cerr << "[RKNN] outputs_get 失败: " << ret << "\n";
        return false;
    }

    out.resize(out_elems_);
    std::memcpy(out.data(), outputs[0].buf, out_elems_ * sizeof(float));
    rknn_outputs_release(ctx_, 1, outputs);
    return true;
}
