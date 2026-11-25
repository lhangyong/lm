# 金属螺母 PatchCore · ONNX 部署包

生成日期：2026-11-25　　源模型：`outputs/models/metal_nut_resnet18_sz256_3x3.pt`（220 张正常样本训练）

> 仓库只收录本说明与转换/自检脚本，ONNX、记忆库、校准图等二进制文件不入库；需要时用 `tools/build_onnx_package.py` 重新打包，或用下文命令重新导出。

## 口径（务必与旧包区分）
| 项 | 本包 | 旧包（outputs/onnx/metal_nut，11-12） |
|---|---|---|
| 记忆库点数 | **2253** | 2294（含泄漏的 224 张模型） |
| IMAGE_THR | **1.4594295025** | 1.464661717414856 |
| PIXEL_THR | **1.4130896330** | 1.4204261302947998 |

## 文件
- `model/metal_nut_patchcore_backbone.onnx`：opset 13，输入 `input` [1,3,256,256]，输出 `patches` [1,1024,384]，权重内嵌
- `model/metal_nut_patchcore_backbone_opset18.onnx`：2026-11-25 用 torch 2.14 重导出，与 opset 13 版特征最大差 2.38e-07，需要高版本 opset 时再换
- `model/memory_bank.npz`：`features`(2253,384) float32、`patch_scores`(2253,)、`grid_size`、`feature_dim`
- `model/memory_bank.bin`：与 npz 的 features 等值的 float32 原始数据，C++ 可直接读
- `model/memory_meta.txt` / `model/onnx_meta.json`：形状、阈值、输入输出契约

## 预处理（必须一致）
RGB、短边缩放到 256 后中心裁剪 256、`pixel/255.0`、均值 0 / 方差 255（即不再做 ImageNet 归一化）。

## 后处理
1. patches 变形为 (1024,384)
2. 每个 patch 对 `memory_bank` 取最近邻欧氏距离（k=1，不需要重加权）
3. 按 32×32 还原成热力图，高斯平滑 σ=4.0（半径 12，零填充）
4. 热力图最大值 = 图像级分数；与 `IMAGE_THR` 比较判 OK/NG
5. 像素级判定用 `PIXEL_THR`

## 指标（板端 RK3588 / NPU，115 张测试集）
缺陷召回 85/93 = 91.4%；good 误报 4/22 = 18.2%；单帧约 56 ms。

## 转 RKNN
```
python convert_rknn.py --onnx model/metal_nut_patchcore_backbone.onnx \
    --platform rk3588 --dtype fp16 --out metal_nut_patchcore_backbone.rknn
```
INT8 需在 x86 Linux + rknn-toolkit2 上跑，量化后用 `calib_images/` 重标阈值。

## 自检
```
python check_onnx.py
```
会打印 ONNX 输入输出形状、记忆库点数与阈值，并跑一次零输入前向确认没有 NaN。

## 本次打包自检结果
- ONNX 特征 vs PyTorch：最大绝对误差 4.053e-06
- 端到端分数 vs PyTorch：最大差 5.960e-07
- ONNX Runtime CPU-EP 单帧 9.60 ms，CUDA 4.68 ms（预热 20 次后 100 次 × 3 轮取中位，见 outputs/onnx/metal_nut/rebuild/bench.txt）
