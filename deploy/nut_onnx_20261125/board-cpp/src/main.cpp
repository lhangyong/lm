// RK3588 金属螺母质检推理主程序。
//
// 用法:
//   qc_detect <model.rknn> <图片或目录> [heatmap_dir]
//
// 资源文件（memory_bank.bin / memory_meta.txt）与 .rknn 放在同一目录。
// 输出: 每张图的异常分数、OK/NG 判定；给了 heatmap_dir 时另存热力图 PNG。
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "memory_bank.h"
#include "postprocess.h"
#include "rknn_model.h"

namespace fs = std::filesystem;

static bool is_image(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static void save_heatmap(const std::vector<float>& heat, int grid,
                         const std::string& out_path) {
    float lo = *std::min_element(heat.begin(), heat.end());
    float hi = *std::max_element(heat.begin(), heat.end());
    cv::Mat m(grid, grid, CV_32F);
    for (int i = 0; i < grid * grid; ++i) m.at<float>(i / grid, i % grid) = heat[i];
    if (hi - lo < 1e-6f) hi = lo + 1e-6f;
    m = (m - lo) / (hi - lo);
    cv::Mat u8;
    m.convertTo(u8, CV_8U, 255.0);
    cv::resize(u8, u8, cv::Size(256, 256), 0, 0, cv::INTER_CUBIC);
    cv::imwrite(out_path, u8);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: qc_detect <model.rknn> <图片或目录> [heatmap_dir]\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string target = argv[2];
    std::string heat_dir = argc >= 4 ? argv[3] : "";

    fs::path res_dir = fs::path(model_path).parent_path();

    MemoryBank bank;
    if (!bank.load(res_dir.string())) {
        std::cerr << "[错误] 记忆库加载失败，目录: " << res_dir << "\n";
        return 2;
    }
    const auto& meta = bank.meta();
    std::cout << "[记忆库] M=" << meta.M << " C=" << meta.C
              << " grid=" << meta.grid << " 图像阈值=" << meta.image_thr << "\n";

    RknnModel model;
    if (!model.load(model_path)) return 3;

    Postprocessor post(meta);

    std::vector<fs::path> images;
    fs::path tgt(target);
    if (fs::is_directory(tgt)) {
        for (const auto& e : fs::directory_iterator(tgt)) {
            if (e.is_regular_file() && is_image(e.path())) images.push_back(e.path());
        }
    } else if (fs::exists(tgt)) {
        images.push_back(tgt);
    } else {
        std::cerr << "[错误] 路径不存在: " << target << "\n";
        return 4;
    }

    if (!heat_dir.empty()) fs::create_directories(heat_dir);

    int ng = 0;
    for (const auto& p : images) {
        cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[跳过] 读图失败: " << p.string() << "\n";
            continue;
        }
        std::vector<float> patches;
        if (!model.infer(img, patches)) {
            std::cerr << "[跳过] 推理失败: " << p.string() << "\n";
            continue;
        }
        DetectResult r = post.run(patches, bank);
        if (r.ng) ++ng;
        std::cout << (r.ng ? "NG" : "OK") << "  score=" << r.score
                  << "  " << p.filename().string() << "\n";

        if (!heat_dir.empty()) {
            save_heatmap(r.heat, r.grid,
                         heat_dir + "/" + p.stem().string() + "_heat.png");
        }
    }
    std::cout << "完成: " << images.size() << " 张, NG " << ng << " 张\n";
    return 0;
}
