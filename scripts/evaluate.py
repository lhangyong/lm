"""评估入口：在测试集上计算 AUROC / PRO / 混淆矩阵，并生成可视化。

用法::

    python scripts/evaluate.py                 # 全量测试集
    python scripts/evaluate.py --limit 6       # 每类抽样 6 张（快速验证）
    python scripts/evaluate.py --threshold 2.5 # 手动指定图像阈值
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from patchcore_qc.config import RUN_DIR, VIS_DIR, load_config
from patchcore_qc.dataset import balance_subset, build_test_items
from patchcore_qc.detector import DefectInspector, Thresholds
from patchcore_qc.metrics import cross_validate_thresholds, evaluate_predictions
from patchcore_qc.viz import montage


def evaluate(cfg, limit: int = 0, threshold: Optional[float] = None, save_visuals: int = 10,
             out_dir: Optional[Path] = None, save: bool = True) -> dict:
    out_dir = Path(out_dir or (RUN_DIR / "eval"))
    vis_dir = VIS_DIR / "eval"
    vis_dir.mkdir(parents=True, exist_ok=True)

    inspector = DefectInspector(cfg)
    if threshold is not None:
        ratio = float(threshold) / max(1e-8, inspector.thresholds.image_score)
        inspector.thresholds = Thresholds(image_score=float(threshold),
                                          pixel_score=inspector.thresholds.pixel_score * ratio,
                                          defect_ratio=cfg.report.defect_ratio_threshold,
                                          source="manual")

    items = build_test_items(cfg)
    if limit:
        items = balance_subset(items, limit)

    model = inspector.model
    print(f"测试样本: {len(items)} 张 | 模型 {cfg.model_path().name} | 设备 {model.device}")
    print(f"阈值 {inspector.thresholds.image_score:.4f}（{inspector.thresholds.source}）")
    t0 = time.time()
    res = model.predict_items(items, verbose=True)
    infer_time = time.time() - t0
    maps = res["maps"][:, 0] if res["maps"].ndim == 4 else res["maps"]
    metrics = evaluate_predictions(items, res["scores"], maps, inspector.thresholds.image_score)
    metrics["inference"] = {"total_s": round(infer_time, 3),
                            "per_image_ms": round(infer_time * 1000 / max(1, len(items)), 2)}
    metrics["threshold_source"] = inspector.thresholds.source
    metrics["cv_best_threshold"] = cross_validate_thresholds(res["scores"],
                                                             np.array([it.label for it in items]))
    metrics["config"] = cfg.to_dict()
    metrics["model_file"] = str(cfg.model_path())

    print("=" * 68)
    print(f"图像级 AUROC : {metrics['image_auroc']:.4f}")
    print(f"像素级 AUROC : {metrics['pixel_auroc']:.4f}")
    print(f"PRO (AUPRO)  : {metrics['aupro']:.4f}    AUPRO@0.3: {metrics['aupro_0_3']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"阈值分类     : Acc={metrics['accuracy']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    print(f"混淆矩阵     : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"单图推理     : {metrics['inference']['per_image_ms']} ms")
    print(f"逐类别 AUROC : " + ", ".join(f"{k}={v['image_auroc']:.4f}" for k, v in metrics["per_defect_auroc"].items()))
    print(f"最优阈值扫描 : {metrics['cv_best_threshold']['threshold']:.4f} "
          f"(balanced_error={metrics['cv_best_threshold']['balanced_error']:.4f})")
    print("=" * 68)

    # 可视化：每类取前 2 张
    by_cls = defaultdict(list)
    for it in items:
        by_cls[it.defect_type].append(it)
    chosen = [x for cls in sorted(by_cls) for x in by_cls[cls][:2]][:save_visuals]
    for it in chosen:
        inspector.inspect_image(it.path, out_dir=vis_dir)
    mt = montage(sorted(vis_dir.glob("*_overlay.png")), max_items=8)
    if mt is not None:
        cv2.imencode(".png", mt)[1].tofile(str(vis_dir / "montage.png"))
        metrics["montage"] = str(vis_dir / "montage.png")

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        print(f"评估结果已保存: {out_dir / 'metrics.json'}")
    return metrics


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="PatchCore 评估")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--limit", type=int, default=0, help="每类抽样上限（0=全量）")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.data_root: cfg.data.root = args.data_root
    if args.dataset: cfg.data.dataset = args.dataset
    if args.device: cfg.model.device = args.device
    evaluate(cfg, limit=args.limit, threshold=args.threshold, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
