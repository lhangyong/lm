"""数据与评估工具层：只读查询数据集结构、已训练模型、最近评估结果。

路径优先取自调用方传入的配置；未配置时使用配置文件里的默认数据集。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from patchcore_qc.config import DATA_ROOT, MODEL_DIR, OUTPUT_ROOT, AppConfig
from patchcore_qc.dataset import list_images
from .tools_detect import pack

_CTX: Dict[str, Any] = {}


def set_context(cfg: AppConfig) -> None:
    _CTX["cfg"] = cfg


def get_cfg() -> Optional[AppConfig]:
    return _CTX.get("cfg")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


@tool
def list_datasets() -> str:
    """列出数据根目录下的数据集（MVTec 风格：train/good、test/*、ground_truth/*）。"""
    cfg = get_cfg()
    root = Path(cfg.data.root) if cfg else DATA_ROOT
    if not root.exists():
        return pack(f"数据根目录不存在: {root}", {"error": "no_data_root", "root": str(root)})
    datasets = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        train, test = d / "train" / "good", d / "test"
        if not (train.exists() or test.exists()):
            continue
        types = sorted(p.name for p in test.iterdir() if p.is_dir()) if test.exists() else []
        datasets.append({"name": d.name, "train": len(list_images(train)),
                         "test_types": types,
                         "test_total": sum(len(list_images(test / t)) for t in types)})
    lines = [f"数据根目录: {root}"] + [
        f"- {d['name']}: 训练 {d['train']} 张，测试 {d['test_total']} 张，类别 {d['test_types']}" for d in datasets]
    return pack("\n".join(lines), {"root": str(root), "datasets": datasets})


@tool
def dataset_overview(dataset: str = "") -> str:
    """统计某个数据集的规模与类别分布（训练/测试/像素级标注），默认当前配置的数据集。

    Args:
        dataset: 数据集目录名，例如 metal_nut；留空使用配置中的数据集
    """
    import cv2
    import numpy as np
    cfg = get_cfg()
    name = dataset or (cfg.data.dataset if cfg else "metal_nut")
    root = Path(cfg.data.root) if cfg else DATA_ROOT
    base = root / name
    if not base.exists():
        return pack(f"数据集不存在: {base}（先用 list_datasets 查看）", {"error": "not_found", "path": str(base)})
    train, test, gt = base / "train" / "good", base / "test", base / "ground_truth"
    dist, total = [], 0
    for d in sorted(p for p in test.iterdir() if p.is_dir()):
        n = len(list_images(d))
        total += n
        dist.append({"class": d.name, "count": n, "has_gt": (gt / d.name).exists()})
    sample = next(iter(list_images(test / "good")), None)
    size = None
    if sample is not None:
        img = cv2.imdecode(np.fromfile(str(sample), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            size = [int(img.shape[1]), int(img.shape[0])]
    payload = {"dataset": name, "train_good": len(list_images(train)), "test_classes": dist,
               "test_total": total, "sample_size": size}
    lines = [f"数据集 {name}: 无监督训练集 train/good {payload['train_good']} 张，测试 {total} 张"]
    lines += [f"  - {d['class']}: {d['count']} 张" + ("（有像素级标注）" if d["has_gt"] else "") for d in dist]
    if size:
        lines.append(f"  图像尺寸示例: {size[0]}x{size[1]}")
    return pack("\n".join(lines), payload)


@tool
def list_models() -> str:
    """查看已训练的 PatchCore 模型文件及其标定阈值。"""
    import json
    files = sorted(MODEL_DIR.glob("*.pt")) if MODEL_DIR.exists() else []
    if not files:
        return pack("还没有训练好的模型，请先运行 python scripts/train.py", {"models": []})
    models = []
    for f in files:
        cal_path = Path(str(f) + ".calibration.json")
        cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
        models.append({"file": f.name, "size": _fmt_size(f.stat().st_size),
                       "image_threshold": cal.get("image_score"), "pixel_threshold": cal.get("pixel_score")})
    lines = [f"- {m['file']}（{m['size']}，图像阈值 {m['image_threshold']}，像素阈值 {m['pixel_threshold']}）"
             for m in models]
    return pack("已训练模型:\n" + "\n".join(lines), {"models": models})


@tool
def read_eval_report() -> str:
    """读取最近一次评估（scripts/evaluate.py）的指标：AUROC、PRO、混淆矩阵、逐类别表现。"""
    import json
    path = OUTPUT_ROOT / "runs" / "eval" / "metrics.json"
    if not path.exists():
        return pack("还没有评估结果，请先运行 python scripts/evaluate.py", {"error": "no_metrics"})
    m = json.loads(path.read_text(encoding="utf-8"))
    cm = m.get("confusion_matrix", {})
    per = ", ".join(f"{k}={v['image_auroc']:.3f}" for k, v in m.get("per_defect_auroc", {}).items())
    lines = [
        f"图像级 AUROC: {m['image_auroc']:.4f}",
        f"像素级 AUROC: {m['pixel_auroc']:.4f}",
        f"PRO/AUPRO: {m['aupro']:.4f}（AUPRO@0.3 = {m['aupro_0_3']:.4f}）",
        f"阈值 {m['threshold']:.4f}（{m.get('threshold_source', '?')}）: Acc={m['accuracy']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}",
        f"混淆矩阵: TN={cm.get('tn')} FP={cm.get('fp')} FN={cm.get('fn')} TP={cm.get('tp')}",
        f"逐类别 AUROC: {per}",
        f"单图推理耗时: {m.get('inference', {}).get('per_image_ms', '?')} ms",
    ]
    payload = {"image_auroc": m["image_auroc"], "pixel_auroc": m["pixel_auroc"], "aupro": m["aupro"],
               "threshold": m["threshold"], "confusion_matrix": cm, "per_defect_auroc": m["per_defect_auroc"],
               "f1": m["f1"], "precision": m["precision"], "recall": m["recall"]}
    return pack("\n".join(lines), payload)


DATA_TOOLS = [list_datasets, dataset_overview, list_models, read_eval_report]
