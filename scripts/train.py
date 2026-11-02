"""训练入口：只用正常样本构建 PatchCore 记忆库，并在训练集上标定阈值。

用法::

    python scripts/train.py                                   # 读取 configs/default.yaml
    python scripts/train.py --image-size 256 --coreset-ratio 0.01
    python scripts/train.py --backbone wide_resnet50_2 --device cuda
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from patchcore_qc.config import load_config
from patchcore_qc.dataset import build_train_items
from patchcore_qc.patchcore import PatchCore


def train(cfg, verbose: bool = True) -> PatchCore:
    torch.manual_seed(0)
    np.random.seed(0)
    model = PatchCore.from_config(cfg)
    if verbose:
        print(f"设备: {model.device} | 数据集: {cfg.dataset_dir}")
        if model.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_items = build_train_items(cfg)
    if verbose:
        print(f"训练正常样本: {len(train_items)} 张（无监督：不使用任何缺陷标注）")

    with torch.no_grad():
        model.fit(train_items, verbose=verbose)
        t0 = time.time()
        cal = model.calibrate(train_items)
        if verbose:
            print(f"标定耗时 {time.time() - t0:.1f}s")

    path = cfg.model_path()
    model.save(path)
    cfg.dump_yaml(path.with_suffix(".config.yaml"))
    Path(str(path) + ".calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    if verbose:
        print(f"模型已保存: {path}")
        print(f"训练集正常分数: mean={cal['train_mean']:.4f} std={cal['train_std']:.4f} "
              f"P99.5={cal['train_p99_5']:.4f} max={cal['train_max']:.4f}")
        print(f"标定阈值: image_score={cal['image_score']:.5f}  pixel_score={cal['pixel_score']:.5f}")
    return model


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="PatchCore 训练（无监督，仅用 train/good）")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--backbone", default=None, choices=["resnet18", "wide_resnet50_2"])
    p.add_argument("--layers", default=None, help="逗号分隔，如 layer2,layer3")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--auto-crop", dest="auto_crop", action="store_true", default=None)
    p.add_argument("--no-auto-crop", dest="auto_crop", action="store_false")
    p.add_argument("--local-pool", dest="local_pool", action="store_true", default=None)
    p.add_argument("--no-local-pool", dest="local_pool", action="store_false")
    p.add_argument("--normalize", dest="normalize", action="store_true", default=None)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--coreset-ratio", type=float, default=None)
    p.add_argument("--max-coreset", type=int, default=None)
    p.add_argument("--n-neighbors", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.dataset:
        cfg.data.dataset = args.dataset
    m = cfg.model
    if args.backbone: m.backbone = args.backbone
    if args.layers: m.layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    if args.image_size:
        m.image_size = m.center_crop = args.image_size
    if args.auto_crop is not None: m.auto_crop = args.auto_crop
    if args.local_pool is not None: m.local_pool = args.local_pool
    if args.normalize is not None: m.normalize = args.normalize
    if args.coreset_ratio is not None: m.coreset_ratio = args.coreset_ratio
    if args.max_coreset: m.max_coreset = args.max_coreset
    if args.n_neighbors: m.n_neighbors = args.n_neighbors
    if args.batch_size: m.batch_size = args.batch_size
    if args.device: m.device = args.device
    train(cfg)


if __name__ == "__main__":
    main()
