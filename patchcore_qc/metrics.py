"""评估：图像级 AUROC、像素级 AUROC、PRO（连通域重叠）、分类混淆矩阵。

- 图像级：score >= 阈值 -> NG；AUROC 与阈值无关。
- 像素级：热力图逐像素与 GT 掩码比较（单类，缺陷为阳性）。
- PRO：Per-Region Overlap，AUPRO_0.3 —— MVTec 官方使用的区域级指标。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

from .dataset import ImgItem


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_score = np.asarray(y_score).reshape(-1).astype(np.float64)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def region_pro(gt_mask: np.ndarray, amap: np.ndarray, max_steps: int = 300) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """单个样本的区域重叠曲线（fpr, pro），参照官方 PRO 实现。

    对预定义的 300 个阈值点 t∈[0,1]，取热力图的前 t 分位数为阈值，
    计算每个 GT 连通域被覆盖的比例，再对所有连通域求平均。
    """
    import cv2
    if gt_mask is None or gt_mask.sum() == 0:
        return None
    _, labels = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    flat_sorted = np.sort(np.asarray(amap).reshape(-1).astype(np.float64))[::-1]
    limits = np.linspace(0, 1, max_steps)
    pro_curve = np.zeros(max_steps, dtype=np.float64)
    n_regions = 0
    for i in range(1, int(labels.max()) + 1):
        region = labels == i
        area = int(region.sum())
        if area == 0:
            continue
        n_regions += 1
        region_sorted = -np.sort(np.asarray(amap)[region].reshape(-1).astype(np.float64))[::-1]
        thrs = np.array([flat_sorted[min(flat_sorted.size - 1, int(round(t * (flat_sorted.size - 1))))]
                         for t in limits])
        # region_sorted 为升序的 -score：searchsorted 直接统计 > thr 的个数
        k = np.searchsorted(region_sorted, -thrs, side="left")
        pro_curve += k / area
    if n_regions == 0:
        return None
    return limits, pro_curve / n_regions


def aupo_curve(curves: Sequence[Tuple[np.ndarray, np.ndarray]], fpr_limit: float = 0.3) -> Dict[str, float]:
    """对多个样本的区域重叠曲线做平均并积分得到 (A)UPRO。"""
    curves = [c for c in curves if c is not None]
    if not curves:
        return {"aupro": float("nan"), "aupro_0_3": float("nan"), "mean_fpr": [], "mean_curves": []}
    fprs = np.concatenate([c[0] for c in curves])
    pros = np.concatenate([c[1] for c in curves])
    # 对齐到 300 个点
    grid = np.linspace(0, 1, 300)
    interp = [np.interp(grid, c[0], c[1]) for c in curves]
    mean = np.mean(interp, axis=0)
    mask = grid <= fpr_limit
    aupro = float(auc(grid[mask], mean[mask]))
    aupro_full = float(auc(grid, mean))
    return {"aupro": aupro_full, "aupro_0_3": aupro,
            "mean_fpr": grid.tolist(), "mean_curves": mean.tolist()}


def evaluate_predictions(items: Sequence[ImgItem], scores: np.ndarray, maps: np.ndarray,
                         threshold: float) -> Dict:
    """items 与推理结果一一对应；maps 尺寸应与预处理后的掩码一致。"""
    y_true = np.array([it.label for it in items])
    scores = np.asarray(scores, dtype=np.float64)

    img_auroc = _safe_auroc(y_true, scores)

    # 逐缺陷类别 AUROC（good vs 某类缺陷）
    per_type: Dict[str, Dict] = {}
    types = sorted({it.defect_type for it in items})
    type_arr = np.array([it.defect_type for it in items])
    for t in types:
        if t == "good":
            continue
        idx = np.where((type_arr == t) | (type_arr == "good"))[0]
        per_type[t] = {
            "n": int((type_arr == t).sum()),
            "image_auroc": _safe_auroc(y_true[idx], scores[idx]),
            "mean_score": float(scores[type_arr == t].mean()),
        }

    # 像素级指标
    pix_y, pix_s = [], []
    pro_curves = []
    rng = np.random.default_rng(0)
    for it, amap in zip(items, maps):
        mk = it.mask
        if mk is None:
            continue
        h, w = mk.shape
        a = amap
        if a.shape != mk.shape:
            import cv2
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        # 每个样本统一用同一个布尔掩码做 30% 像素权重采样（兼顾精度与内存）
        sel = rng.random(mk.shape) < 0.3
        pix_y.append((mk > 0.5).astype(np.uint8)[sel])
        pix_s.append(a[sel])
        if mk.sum() > 0:
            pro_curves.append(region_pro(mk, a))
    if pix_y:
        pix_y_arr = np.concatenate(pix_y)
        pix_s_arr = np.concatenate(pix_s)
        pix_auroc = _safe_auroc(pix_y_arr, pix_s_arr)
        pro = aupo_curve(pro_curves)
    else:
        pix_auroc, pro = float("nan"), {"aupro": float("nan"), "aupro_0_3": float("nan"),
                                        "mean_fpr": [], "mean_curves": []}

    # 阈值下的分类指标
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    fpr_val = fp / max(1, fp + tn)
    fnr_val = fn / max(1, fn + tp)

    # 最佳 F1 阈值（阈值扫描），用于和固定阈值形成对比
    best_f1, best_thr = 0.0, float(threshold)
    if len(np.unique(y_true)) > 1:
        for th in np.unique(scores):
            p = (scores >= th).astype(int)
            if p.sum() == 0:
                continue
            f = precision_recall_fscore_support(y_true, p, average="binary", zero_division=0)[2]
            if f > best_f1:
                best_f1, best_thr = float(f), float(th)
    fpr_c, tpr_c = _roc(y_true, scores)

    # 阈值灵敏度：在当前阈值上下浮动时 P/R/F1 的变化
    sweep = []
    if len(np.unique(y_true)) > 1:
        for delta in (0.0, 1.0, 2.0, 3.0, 5.0):
            th = float(threshold) * (1 + delta / 100)
            p = (scores >= th).astype(int)
            if p.sum() == 0:
                sweep.append({"delta_pct": delta, "threshold": round(th, 5), "precision": 0.0,
                              "recall": 0.0, "f1": 0.0, "ng": 0})
                continue
            pr_, rc_, f1_, _ = precision_recall_fscore_support(y_true, p, average="binary", zero_division=0)
            sweep.append({"delta_pct": delta, "threshold": round(th, 5), "precision": round(float(pr_), 4),
                          "recall": round(float(rc_), 4), "f1": round(float(f1_), 4), "ng": int(p.sum())})

    result = {
        "image_auroc": img_auroc,
        "pixel_auroc": pix_auroc,
        "aupro": pro["aupro"],
        "aupro_0_3": pro["aupro_0_3"],
        "per_defect_auroc": per_type,
        "threshold": float(threshold),
        "threshold_sweep": sweep,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "false_positive_rate": float(fpr_val), "false_negative_rate": float(fnr_val),
        "best_f1": best_f1, "best_threshold": best_thr,
        "roc": {"fpr": fpr_c.tolist() if len(fpr_c) else [], "tpr": tpr_c.tolist() if len(tpr_c) else []},
        "pro_curve": {"fpr": pro["mean_fpr"], "pro": pro["mean_curves"]},
        "counts": {"total": int(len(items)), "ng_true": int(y_true.sum()),
                   "ok_true": int((1 - y_true).sum())},
        "images": [
            {"name": it.path.name, "defect_type": it.defect_type, "label": int(it.label),
             "score": float(s), "pred": int(p >= threshold), "correct": bool(int(it.label) == int(p >= threshold))}
            for it, s, p in zip(items, scores, scores)
        ],
    }
    return result


def _roc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(np.unique(y_true)) < 2:
        return np.array([]), np.array([])
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    return fpr, tpr


def cross_validate_thresholds(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """给定图像分数与真值，扫描出使平衡错误率（FPR+FNR）最低的阈值。

    仅用于离线分析「最优阈值在哪里」，线上阈值来源见 detector.resolve_thresholds。
    """
    ok_scores = scores[labels == 0]
    ng_scores = scores[labels == 1]
    if ok_scores.size == 0 or ng_scores.size == 0:
        return {"threshold": float(np.percentile(scores, 80)) if scores.size else 0.5,
                "balanced_error": float("nan")}
    candidates = np.unique(np.concatenate([ok_scores, ng_scores]))
    best, best_err = float(candidates[0]), float("inf")
    for t in candidates:
        err = ((ok_scores >= t).mean() + (ng_scores < t).mean()) / 2
        if err < best_err:
            best_err, best = float(err), float(t)
    return {"threshold": best, "balanced_error": best_err}
