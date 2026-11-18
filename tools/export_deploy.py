# -*- coding: utf-8 -*-
"""PatchCore 特征骨干重导出 + 校验（与 board 部署包同口径）

工单编号：人工智能CV-智能交通路口管理-嵌入式设备推理加速、部署升级任务

为什么单独有这个脚本：
    `outputs/onnx/metal_nut/` 里的部署包是 11-12 产出的，当时用的是一个临时脚本，
    仓库里没有留下可复现的入口。这个脚本把那一份的导出口径固化下来：
    输入名 input、输出名 patches、形状 [1,1024,384]、权重内嵌不拆 .data 文件。

它会做三件事：
    1) 从训练好的 PatchCore 权重重导出骨干 ONNX（默认写到 rebuild/，不覆盖正式包）
    2) 与 PyTorch 原模型比对特征，最大绝对误差必须 < 1e-3
    3) 与正式包里的 ONNX 比对，确认两者等价

用法：
    python tools/export_deploy.py                # 只校验，产物写 rebuild/
    python tools/export_deploy.py --replace      # 校验通过后覆盖正式包（自动备份）
"""
import os
import sys
import json
import time
import shutil
import argparse
import statistics
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODEL = os.path.join(ROOT, 'outputs', 'models', 'metal_nut_resnet18_sz256_3x3.pt')
PKG = os.path.join(ROOT, 'outputs', 'onnx', 'metal_nut')
OFFICIAL = os.path.join(PKG, 'metal_nut_patchcore_backbone.onnx')


def build_and_export(patch_embed, image_size, out_path, opset=13):
    patch_embed = patch_embed.eval().cpu()
    dummy = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        ref = patch_embed(dummy).cpu().numpy()
    torch.onnx.export(
        patch_embed, dummy, out_path,
        input_names=['input'], output_names=['patches'],
        opset_version=opset, do_constant_folding=True, dynamic_axes=None,
        external_data=False)
    return ref


def run_onnx(path, shape):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: np.zeros(shape, dtype=np.float32)})[0]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    return out, outs, sess.get_providers()


def bench(path, shape, n=100, rounds=3, providers=None):
    """返回 (中位单帧毫秒, 实际生效的执行提供者)。CUDA 加载失败会静默回落 CPU，必须看 get_providers()。"""
    import onnxruntime as ort
    if providers is None:
        providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(path, providers=providers)
    x = np.zeros(shape, dtype=np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(20):
        sess.run(None, {name: x})
    res = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {name: x})
        res.append((time.perf_counter() - t0) / n * 1e3)
    return statistics.median(res), sess.get_providers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replace', action='store_true', help='校验通过后覆盖正式包里的 ONNX（先备份）')
    ap.add_argument('--opset', type=int, default=13)
    args = ap.parse_args()

    from patchcore_qc.patchcore import PatchCore

    model = PatchCore.load(MODEL, device='cpu')
    print('模型: backbone=%s layers=%s image_size=%d feature_dim=%d grid=%d'
          % (model.backbone, model.layers, model.image_size, model.feature_dim, model.grid_size))

    rebuild = os.path.join(PKG, 'rebuild')
    os.makedirs(rebuild, exist_ok=True)
    tmp_onnx = os.path.join(rebuild, 'metal_nut_patchcore_backbone.onnx')

    ref = build_and_export(model.patch_embed, model.image_size, tmp_onnx, args.opset)
    size_mb = os.path.getsize(tmp_onnx) / 1024 / 1024
    print('重导出: %s  %.2f MB' % (tmp_onnx, size_mb))

    shape = (1, 3, model.image_size, model.image_size)
    out_new, outs, prov = run_onnx(tmp_onnx, shape)
    print('输出节点: %s  形状 %s' % (outs[0][0], outs[0][1]))
    d_pt = float(np.max(np.abs(out_new - ref)))
    print('vs PyTorch 最大绝对误差: %.3e  ->  %s' % (d_pt, '通过' if d_pt < 1e-3 else '不通过'))

    d_pkg = None
    if os.path.exists(OFFICIAL):
        out_old, outs_old, _ = run_onnx(OFFICIAL, shape)
        d_pkg = float(np.max(np.abs(out_new - out_old)))
        print('vs 正式包 ONNX（%s）最大绝对误差: %.3e  ->  %s'
              % (outs_old[0][0], d_pkg, '等价' if d_pkg < 1e-3 else '不等价'))

    ms, prov = bench(tmp_onnx, shape, providers=['CPUExecutionProvider'])
    print('ONNX Runtime(CPU-EP) 单帧 %.2f ms（%.0f FPS 上限）' % (ms, 1000 / ms))
    ms_gpu, prov_gpu = bench(tmp_onnx, shape, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    gpu_on = 'CUDAExecutionProvider' in prov_gpu
    if gpu_on:
        print('ONNX Runtime(CUDA)  单帧 %.2f ms（%.0f FPS 上限）' % (ms_gpu, 1000 / ms_gpu))
    else:
        print('CUDA 未加载（已回落 CPU，%.2f ms）；如需要先把 torch\\lib 加入 PATH' % ms_gpu)

    ok = d_pt < 1e-3 and (d_pkg is None or d_pkg < 1e-3)
    if args.replace and ok:
        if os.path.exists(OFFICIAL):
            bak = os.path.join(PKG, 'metal_nut_patchcore_backbone_20261112_backup.onnx')
            if not os.path.exists(bak):
                shutil.copy2(OFFICIAL, bak)
                print('已备份原包:', os.path.basename(bak))
        shutil.copy2(tmp_onnx, OFFICIAL)
        print('已覆盖正式包 ONNX（口径一致，节点名保持 input / patches）')
    elif args.replace:
        print('[跳过覆盖] 校验未通过，正式包保持原样')

    with open(os.path.join(rebuild, 'bench.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join([
            'PatchCore 骨干（resnet18，256×256，layer2+layer3，3x3 邻域池化）',
            'ONNX Runtime CPU-EP 单帧 %.2f ms（%.0f FPS 上限）' % (ms, 1000 / ms),
            ('ONNX Runtime CUDA   单帧 %.2f ms（%.0f FPS 上限）' % (ms_gpu, 1000 / ms_gpu))
            if gpu_on else 'CUDA 提供者未加载，已回落 CPU',
            'vs PyTorch 最大绝对误差 %.3e' % d_pt,
            ('vs 正式包最大绝对误差 %.3e' % d_pkg) if d_pkg is not None else '正式包不存在，跳过比对',
            'CPU-EP 实际提供者 %s；CUDA 测试实际提供者 %s' % (prov[0], ','.join(prov_gpu)),
            '测法 100 次 × 3 轮取中位，预热 20 次；CUDA 需先 import torch（torch\\lib 内自带 cublasLt64_12.dll 与 cuDNN 9）',
            '记忆库 %d×%d' % (model.memory.features.shape[0], model.feature_dim),
        ]) + '\n')
    print('记录已写入:', os.path.join(rebuild, 'bench.txt'))
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
