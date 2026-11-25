#pragma once
// 记忆库加载：读取 memory_bank.bin（features + patch_scores，float32 行优先）
// 和 memory_meta.txt（key=value 超参/阈值）。
#include <string>
#include <vector>

struct BankMeta {
    int M = 0;            // core patch 数量
    int C = 0;            // 特征维度
    int grid = 0;         // patch 网格边长（32）
    int image_size = 256;
    float sigma = 4.0f;
    int n_neighbors = 1;
    float image_thr = 0.0f;
    float pixel_thr = 0.0f;
};

class MemoryBank {
public:
    bool load(const std::string& dir);

    const std::vector<float>& features() const { return features_; }
    const std::vector<float>& patch_scores() const { return patch_scores_; }
    const BankMeta& meta() const { return meta_; }

private:
    std::vector<float> features_;      // M * C，行优先
    std::vector<float> patch_scores_;   // M
    BankMeta meta_;
};
