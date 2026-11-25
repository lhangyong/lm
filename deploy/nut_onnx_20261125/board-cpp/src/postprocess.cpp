#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr float kEps = 1e-4f;

float dot(const float* a, const float* b, int c) {
    float s = 0.f;
    for (int i = 0; i < c; ++i) s += a[i] * b[i];
    return s;
}
}  // namespace

Postprocessor::Postprocessor(const BankMeta& meta) : meta_(meta) {
    radius_ = std::max(1, static_cast<int>(std::round(3.f * meta_.sigma)));
    const int k = 2 * radius_ + 1;
    gauss1d_.resize(k);
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        int x = i - radius_;
        float v = std::exp(-static_cast<float>(x * x) / (2.f * meta_.sigma * meta_.sigma));
        gauss1d_[i] = v;
        sum += v;
    }
    for (auto& v : gauss1d_) v /= sum;
}

DetectResult Postprocessor::run(const std::vector<float>& patches,
                                const MemoryBank& bank) const {
    const int M = meta_.M;
    const int C = meta_.C;
    const int grid = meta_.grid;
    const int N = grid * grid;
    const auto& feats = bank.features();

    // 预计算每个 core 向量的模长平方
    std::vector<float> core_norm2(M, 0.f);
    for (int j = 0; j < M; ++j) {
        const float* y = feats.data() + static_cast<size_t>(j) * C;
        core_norm2[j] = dot(y, y, C);
    }

    // 每个 patch 到记忆库的最近邻距离（k=1）
    std::vector<float> dist(N, 0.f);
    for (int i = 0; i < N; ++i) {
        const float* x = patches.data() + static_cast<size_t>(i) * C;
        float xn2 = dot(x, x, C);
        float best = std::numeric_limits<float>::infinity();
        for (int j = 0; j < M; ++j) {
            const float* y = feats.data() + static_cast<size_t>(j) * C;
            float d2 = xn2 + core_norm2[j] - 2.f * dot(x, y, C);
            if (d2 < 0.f) d2 = 0.f;
            if (d2 < best) best = d2;
        }
        dist[i] = std::sqrt(std::max(best, kEps));
    }

    // 还原 grid×grid 热力图，做可分离高斯平滑（零填充，与原工程 conv2d padding 一致）
    std::vector<float> smooth(N, 0.f);
    std::vector<float> tmp(N, 0.f);

    // 纵向卷积
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            float acc = 0.f;
            for (int k = 0; k < static_cast<int>(gauss1d_.size()); ++k) {
                int rr = r - radius_ + k;
                if (rr < 0 || rr >= grid) continue;
                acc += gauss1d_[k] * dist[rr * grid + c];
            }
            tmp[r * grid + c] = acc;
        }
    }
    // 横向卷积
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            float acc = 0.f;
            for (int k = 0; k < static_cast<int>(gauss1d_.size()); ++k) {
                int cc = c - radius_ + k;
                if (cc < 0 || cc >= grid) continue;
                acc += gauss1d_[k] * tmp[r * grid + cc];
            }
            smooth[r * grid + c] = acc;
        }
    }

    float score = *std::max_element(smooth.begin(), smooth.end());
    DetectResult res;
    res.score = score;
    res.ng = score >= meta_.image_thr;
    res.heat = std::move(smooth);
    res.grid = grid;
    return res;
}
