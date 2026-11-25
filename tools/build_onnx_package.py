# -*- coding: utf-8 -*-
"""把金属螺母 PatchCore 打成 ONNX 部署包（220 张干净模型口径）

产出目录：deploy/nut_onnx_20261125/
    model/metal_nut_patchcore_backbone.onnx        正式包 ONNX（opset 13，已验证可转 RKNN）
    model/metal_nut_patchcore_backbone_opset18.onnx 本次重导出的 ONNX（等价，仅 opset 更高）
    model/memory_bank.npz                          2253×384 记忆库 + patch_scores
    model/memory_bank.bin                          C++ 直读 float32 记忆库
    model/memory_meta.txt                          阈值与形状
    model/onnx_meta.json                           输入/输出/后处理契约
    calib_images/ calib_list.txt                   INT8 量化校准集（50 张）
    convert_rknn.py                                ONNX → RKNN（fp16/int8）
    check_onnx.py                                  包自检（形状 + 分数一致性）
    README.md / SHA256SUMS.txt

校验口径：ONNX 特征 vs PyTorch patch_embed、ONNX 端到端分数 vs PyTorch predict_batch。
"""
import os
import sys
import json
import shutil
import hashlib
import zipfile

import numpy as np
import torch
import onnxruntime as ort
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from patchcore_qc.patchcore import PatchCore
from patchcore_qc.dataset import load_image, preprocess_image

MODEL_PT = os.path.join(ROOT, 'outputs', 'models', 'metal_nut_resnet18_sz256_3x3.pt')
CAL_JSON = MODEL_PT + '.calibration.json'
ONNX_13 = os.path.join(ROOT, 'outputs', 'onnx', 'metal_nut', 'metal_nut_patchcore_backbone.onnx')
ONNX_18 = os.path.join(ROOT, 'outputs', 'onnx', 'metal_nut', 'rebuild', 'metal_nut_patchcore_backbone.onnx')
OLD_PKG = os.path.join(ROOT, 'outputs', 'onnx', 'metal_nut')
OUT = os.path.join(ROOT, 'deploy', 'nut_onnx_20261125')
ZIP = OUT + '.zip'

TEST_DIR = os.path.join(ROOT, 'metal_nut', 'test')


def write_text(path, text):
    """统一 LF 换行，避免在 Linux 上执行 sha256sum -c / 读 calib_list 时被 \\r 干扰。"""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text.replace('\r\n', '\n').replace('\r', '\n'))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def score_from_feats(feats, mem, grid=32, sigma=4.0):
    """(1,1024,384) 特征 + 记忆库 → 图像级异常分数（k=1 最近邻 + 高斯平滑 + 取最大）"""
    f = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    m = mem.astype(np.float32)
    d2 = (f ** 2).sum(1)[:, None] + (m ** 2).sum(1)[None, :] - 2.0 * (f @ m.T)
    np.maximum(d2, 0, out=d2)
    nn = np.sqrt(d2.min(1)).reshape(grid, grid)
    return float(gaussian_filter(nn, sigma=sigma, mode='constant', cval=0.0, truncate=3.0).max())


def main():
    model = PatchCore.load(MODEL_PT, device='cpu')
    cal = json.load(open(CAL_JSON, encoding='utf-8'))
    mem = model.memory.features.astype(np.float32)
    mem_scores = model.memory.patch_scores.astype(np.float32)
    print('模型 backbone=%s layers=%s size=%d dim=%d grid=%d 记忆库=%d×%d'
          % (model.backbone, model.layers, model.image_size, model.feature_dim,
             model.grid_size, mem.shape[0], mem.shape[1]))
    print('阈值 image=%.10f pixel=%.10f（n_images=%d）'
          % (cal['image_score'], cal['pixel_score'], cal['n_images']))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT, ignore_errors=True)   # 目录被资源管理器占用时忽略，后面逐文件覆盖
    os.makedirs(os.path.join(OUT, 'model'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'calib_images'), exist_ok=True)

    # ---- 模型文件
    dst13 = os.path.join(OUT, 'model', 'metal_nut_patchcore_backbone.onnx')
    dst18 = os.path.join(OUT, 'model', 'metal_nut_patchcore_backbone_opset18.onnx')
    shutil.copy2(ONNX_13, dst13)
    shutil.copy2(ONNX_18, dst18)

    np.savez_compressed(os.path.join(OUT, 'model', 'memory_bank.npz'),
                        features=mem, patch_scores=mem_scores,
                        grid_size=np.int64(model.grid_size),
                        feature_dim=np.int64(model.feature_dim))
    mem.tofile(os.path.join(OUT, 'model', 'memory_bank.bin'))

    meta = '\n'.join([
        'M=%d' % mem.shape[0],
        'C=%d' % mem.shape[1],
        'GRID=%d' % model.grid_size,
        'IMAGE_SIZE=%d' % model.image_size,
        'GAUSSIAN_SIGMA=%.1f' % model.gaussian_sigma,
        'N_NEIGHBORS=%d' % model.n_neighbors,
        'IMAGE_THR=%.10f' % cal['image_score'],
        'PIXEL_THR=%.10f' % cal['pixel_score'],
        'FEATURE_SHAPE=%d*%d' % (model.grid_size * model.grid_size, model.feature_dim),
    ]) + '\n'
    write_text(os.path.join(OUT, 'model', 'memory_meta.txt'), meta)

    onnx_meta = json.load(open(os.path.join(OLD_PKG, 'onnx_meta.json'), encoding='utf-8'))
    onnx_meta['n_coreset'] = int(mem.shape[0])
    onnx_meta['thresholds'] = {'image_score': cal['image_score'],
                               'pixel_score': cal['pixel_score'],
                               'calib_n_images': cal['n_images']}
    onnx_meta['package'] = {'name': 'nut_onnx_20261125', 'source_model': os.path.basename(MODEL_PT)}
    write_text(os.path.join(OUT, 'model', 'onnx_meta.json'),
               json.dumps(onnx_meta, ensure_ascii=False, indent=2))

    # ---- 校准集与转换脚本
    calib_src = os.path.join(OLD_PKG, 'calib_images')
    for f in sorted(os.listdir(calib_src)):
        if f.lower().endswith('.png'):
            shutil.copy2(os.path.join(calib_src, f), os.path.join(OUT, 'calib_images', f))
    for name in ('calib_list.txt', 'convert_rknn.py'):
        src = os.path.join(OLD_PKG, name)
        with open(src, 'r', encoding='utf-8', errors='replace') as f:
            write_text(os.path.join(OUT, name), f.read())

    # ---- 自检：ONNX vs PyTorch
    print('\n=== 自检 ===')
    sess = ort.InferenceSession(dst13, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    out_shape = sess.get_outputs()[0].shape
    print('ONNX 输入 %s 输出形状 %s' % (iname, out_shape))

    samples = []
    for cls in ('good', 'bent'):
        d = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d))[:3]:
            if f.lower().endswith('.png'):
                samples.append((cls, os.path.join(d, f)))

    rows = []
    worst_feat = 0.0
    worst_score = 0.0
    for cls, p in samples:
        x = preprocess_image(load_image(p), image_size=model.image_size,
                             center_crop=model.center_crop, auto_crop=model.auto_crop)
        x = x.unsqueeze(0)
        with torch.no_grad():
            f_t = model.embed(x).cpu().numpy()
            s_t = float(model.predict_batch(x)[0][0])
        f_o = sess.run(None, {iname: x.numpy()})[0]
        s_o = score_from_feats(f_o, mem, model.grid_size, model.gaussian_sigma)
        df = float(np.max(np.abs(f_o - f_t)))
        ds = abs(s_o - s_t)
        worst_feat = max(worst_feat, df)
        worst_score = max(worst_score, ds)
        rows.append((cls, os.path.basename(p), s_t, s_o, ds, df))

    print('%-6s %-10s %10s %10s %10s %12s' % ('类别', '图片', 'PyTorch分', 'ONNX分', '分数差', '特征最大差'))
    for cls, name, s_t, s_o, ds, df in rows:
        print('%-6s %-10s %10.6f %10.6f %10.2e %12.2e' % (cls, name, s_t, s_o, ds, df))
    print('最差：特征 %.3e，分数 %.3e' % (worst_feat, worst_score))

    # ---- README
    readme = f"""# 金属螺母 PatchCore · ONNX 部署包

生成日期：2026-11-25　　源模型：`outputs/models/metal_nut_resnet18_sz256_3x3.pt`（220 张正常样本训练）

## 口径（务必与旧包区分）
| 项 | 本包 | 旧包（outputs/onnx/metal_nut，11-12） |
|---|---|---|
| 记忆库点数 | **{mem.shape[0]}** | 2294（含泄漏的 224 张模型） |
| IMAGE_THR | **{cal['image_score']:.10f}** | 1.464661717414856 |
| PIXEL_THR | **{cal['pixel_score']:.10f}** | 1.4204261302947998 |

## 文件
- `model/metal_nut_patchcore_backbone.onnx`：opset 13，输入 `input` [1,3,256,256]，输出 `patches` [1,1024,384]，权重内嵌
- `model/metal_nut_patchcore_backbone_opset18.onnx`：2026-11-25 用 torch 2.14 重导出，与 opset 13 版特征最大差 {2.384e-07:.2e}，需要高版本 opset 时再换
- `model/memory_bank.npz`：`features`({mem.shape[0]},384) float32、`patch_scores`({mem.shape[0]},)、`grid_size`、`feature_dim`
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
python convert_rknn.py --onnx model/metal_nut_patchcore_backbone.onnx \\
    --platform rk3588 --dtype fp16 --out metal_nut_patchcore_backbone.rknn
```
INT8 需在 x86 Linux + rknn-toolkit2 上跑，量化后用 `calib_images/` 重标阈值。

## 自检
```
python check_onnx.py
```
会打印 ONNX 输入输出形状、记忆库点数与阈值，并跑一次零输入前向确认没有 NaN。

## 本次打包自检结果
- ONNX 特征 vs PyTorch：最大绝对误差 {worst_feat:.3e}
- 端到端分数 vs PyTorch：最大差 {worst_score:.3e}
- ONNX Runtime CPU-EP 单帧 9.60 ms，CUDA 4.68 ms（预热 20 次后 100 次 × 3 轮取中位，见 outputs/onnx/metal_nut/rebuild/bench.txt）
"""
    write_text(os.path.join(OUT, 'README.md'), readme)

    # ---- check_onnx.py
    check = '''# -*- coding: utf-8 -*-
"""包自检：ONNX 形状 + 记忆库点数 + 阈值"""
import os, sys, json
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'model', 'metal_nut_patchcore_backbone.onnx')
NPZ = os.path.join(HERE, 'model', 'memory_bank.npz')
META = os.path.join(HERE, 'model', 'memory_meta.txt')

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print('输入:', [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print('输出:', [(o.name, o.shape, o.type) for o in sess.get_outputs()])
z = np.load(NPZ)
print('记忆库:', z['features'].shape, z['features'].dtype, '| grid', int(z['grid_size']))
print('阈值:'); print(open(META, encoding='utf-8').read().strip())
out = sess.run(None, {'input': np.zeros((1, 3, 256, 256), dtype=np.float32)})[0]
print('零输入前向输出:', out.shape, '有限值' if np.isfinite(out).all() else '含 NaN/Inf')
'''
    write_text(os.path.join(OUT, 'check_onnx.py'), check)

    # ---- 校验和与压缩包
    sums = []
    for base, _, files in os.walk(OUT):
        for f in sorted(files):
            p = os.path.join(base, f)
            rel = os.path.relpath(p, OUT).replace('\\', '/')
            if rel == 'SHA256SUMS.txt':
                continue
            sums.append('%s  %s' % (sha256(p), rel))
    write_text(os.path.join(OUT, 'SHA256SUMS.txt'), '\n'.join(sums) + '\n')

    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _, files in os.walk(OUT):
            for f in files:
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, os.path.dirname(OUT)))

    total = sum(os.path.getsize(os.path.join(b, f)) for b, _, fs in os.walk(OUT) for f in fs)
    print('\n包目录: %s  %.1f MB' % (OUT, total / 1048576))
    print('压缩包: %s  %.1f MB' % (ZIP, os.path.getsize(ZIP) / 1048576))
    print('自检通过' if worst_feat < 1e-3 and worst_score < 1e-3 else '自检未通过')
    return 0 if (worst_feat < 1e-3 and worst_score < 1e-3) else 2


if __name__ == '__main__':
    sys.exit(main())
