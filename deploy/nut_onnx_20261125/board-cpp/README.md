# RK3588 板子侧 C++ 推理工程

在 RK3588 上跑金属螺母外观质检：NPU 跑特征主干（`.rknn`），CPU 跑记忆库最近邻比对和判级。

## 一、目录结构

```text
board-cpp/
├── CMakeLists.txt
├── memory_bank.bin      # 记忆库向量（C++ 直接读取，随模型部署）
├── memory_meta.txt      # 维度、网格、平滑参数、判级阈值
├── include/
│   ├── rknn_model.h     # RKNN 模型加载/推理封装
│   ├── memory_bank.h     # 记忆库读取
│   └── postprocess.h     # kNN 距离 + 高斯平滑 + 判级
└── src/
    ├── main.cpp          # 入口：读图 -> 推理 -> 判级 -> 打印/存热力图
    ├── rknn_model.cpp
    ├── memory_bank.cpp
    └── postprocess.cpp
```

## 二、前置：先拿到 `.rknn`

本工程只负责板子上推理，**不包含模型转换**。`.rknn` 需要在装有 rknn-toolkit2 的 Linux 机器上，用上级目录的 `convert_rknn.py` 把 ONNX 转出来（INT8 量化，校准图在 `../calib_images/`）：

```bash
python ../convert_rknn.py --platform rk3588
# 产出 metal_nut_patchcore_backbone.rknn
```

## 三、板子依赖

- OpenCV（`libopencv-dev`，读图/缩放/存热力图）
- RKNN 运行库 `librknnrt.so`（你板子上已装）
- 头文件 `rknn_api.h`：在 RKNN SDK 的 `include/` 里。若板子 `/usr/include` 下没有，把它拷到本工程 `include/` 目录，或编译时用 `-DRKNN_DIR=<SDK根目录>` 指定。

确认运行库在 PATH 能找到：

```bash
ls /usr/lib/librknnrt.so    # 或 /usr/lib/aarch64-linux-gnu/
```

## 四、编译

### 方式 A：板子上本地编译（aarch64）

```bash
cd board-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

若 `rknn_api.h` 不在默认路径：

```bash
cmake .. -DRKNN_DIR=/path/to/rknn_sdk
```

### 方式 B：交叉编译（PC 上编出 ARM 二进制）

工具链用 RKNN SDK 里的 `aarch64-linux-gnu` 交叉编译器，CMake 加 `-DCMAKE_TOOLCHAIN_FILE=<SDK>/toolchain.cmake`，并把 OpenCV 换成 ARM 版；其余同上。

编译产物：`build/qc_detect`。

## 五、部署布局

把下面几样放到板子同一目录（记忆库按约定与 `.rknn` 同目录）：

```text
部署目录/
├── qc_detect                    # 编译出的可执行文件
├── metal_nut_patchcore_backbone.rknn
├── memory_bank.bin
└── memory_meta.txt
```

## 六、运行

```bash
# 单张图
./qc_detect metal_nut_patchcore_backbone.rknn /path/to/nut.jpg

# 整个目录，同时把热力图存到 heat/
./qc_detect metal_nut_patchcore_backbone.rknn /path/to/images heat/
```

输出示例：

```text
[记忆库] M=2253 C=384 grid=32 图像阈值=1.459429
[RKNN] 加载成功 输入 256x256 输出元素数 393216
OK  score=1.40117  000.png
NG  score=1.66873  001.png
完成: 2 张, NG 1 张
```

- `score`：热力图最大值（异常分数）；≥ `memory_meta.txt` 里的 `IMAGE_THR` 判 NG。
- 给了热力图目录时，每张图存一张 `*_heat.png`（0~255 归一化灰度图）。

## 七、口径与注意事项

- 输入：RGB，256×256，像素 0~255 交给 NPU，归一化（/255）在模型内部完成（转换时 mean=0、std=255）。
- 输出 `patches`：1024×384（32×32 个 patch，每块 384 维）。
- 后处理：每个 patch 到 2253 个记忆库向量取最近邻欧氏距离，重排成 32×32 热力图，高斯平滑（sigma=4.0），取最大值为图像分数。该口径已在 PC 上与原 PyTorch 模型逐图核对，分数误差约 1e-7。
- 本目录的 `memory_bank.bin` / `memory_meta.txt` 已与 `../model/` 的干净模型（220 张训练、2253 个核心点、IMAGE_THR=1.459429）对齐；换模型时两处要一起替换，不要混用旧版 2294 点记忆库。
- 训练/测试原图为方形，代码直接缩放到 256×256；若实拍是非方形图，需在预处理里补「缩放短边 + 中心裁剪 256」逻辑，与原工程 `preprocess_image` 对齐。
- INT8 量化后分数尺度可能略有漂移，上线前建议用板子实测一批正常件/缺陷件，必要时按 `memory_meta.txt` 里的 `IMAGE_THR` 重新核阈值。
- 几何量测（内径/对边/壁厚等）是独立 OpenCV 旁路，不在本工程内。
