#include "memory_bank.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

bool MemoryBank::load(const std::string& dir) {
    // ---- 元数据 key=value ----
    std::ifstream mf(dir + "/memory_meta.txt");
    if (!mf) return false;
    std::string line;
    while (std::getline(mf, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string k = line.substr(0, pos);
        std::string v = line.substr(pos + 1);
        if (k == "M") meta_.M = std::stoi(v);
        else if (k == "C") meta_.C = std::stoi(v);
        else if (k == "GRID") meta_.grid = std::stoi(v);
        else if (k == "IMAGE_SIZE") meta_.image_size = std::stoi(v);
        else if (k == "GAUSSIAN_SIGMA") meta_.sigma = std::stof(v);
        else if (k == "N_NEIGHBORS") meta_.n_neighbors = std::stoi(v);
        else if (k == "IMAGE_THR") meta_.image_thr = std::stof(v);
        else if (k == "PIXEL_THR") meta_.pixel_thr = std::stof(v);
    }
    if (meta_.M <= 0 || meta_.C <= 0) return false;

    // ---- 二进制：features(M*C) 后 patch_scores(M)，float32 小端 ----
    std::ifstream bf(dir + "/memory_bank.bin", std::ios::binary);
    if (!bf) return false;
    features_.resize(static_cast<size_t>(meta_.M) * meta_.C);
    patch_scores_.resize(meta_.M);
    bf.read(reinterpret_cast<char*>(features_.data()),
            features_.size() * sizeof(float));
    bf.read(reinterpret_cast<char*>(patch_scores_.data()),
            patch_scores_.size() * sizeof(float));
    return bf.good();
}
