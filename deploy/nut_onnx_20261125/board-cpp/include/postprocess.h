#pragma once
// 板子侧后处理：patch 描述子 -> 记忆库最近邻距离 -> 高斯平滑热力图 -> 图像分数/判级。
// 口径与原 PatchCore 工程一致（n_neighbors=1，欧氏距离，可分离高斯）。
#include <string>
#include <vector>

#include "memory_bank.h"

struct DetectResult {
    float score = 0.f;       // 图像级异常分数（热力图最大值）
    bool ng = false;         // 是否超图像阈值
    std::vector<float> heat;  // grid*grid 平滑后热力图
    int grid = 0;
};

class Postprocessor {
public:
    explicit Postprocessor(const BankMeta& meta);

    // patches: [N, C]，N = grid*grid，行优先
    DetectResult run(const std::vector<float>& patches, const MemoryBank& bank) const;

private:
    BankMeta meta_;
    std::vector<float> gauss1d_;
    int radius_ = 0;
};
