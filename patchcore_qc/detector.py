"""单图/批量缺陷检测器：加载 PatchCore 模型，输出结构化 JSON 结果。

职责：
- 阈值选择（人工设置 > 训练集自动标定 > 配置默认）；
- 缺陷区域几何后处理与分级（Major / Minor / Trace）；
- 可视化素材（叠加 PNG、热力图 SVG、分数直方图 SVG）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import AppConfig
from .dataset import (ImgItem, build_test_groups, build_train_items, iter_batches, list_images,
                      load_image, preprocess_image)
from .patchcore import PatchCore, describe_severity, postprocess
from .viz import heatmap_overlay_svg, mask_overlay_png, score_histogram_svg

# 缺陷类别 -> 工业语义（仅用于报告解释，不参与判定）
DEFECT_SEMANTICS: Dict[str, str] = {
    "bent": "弯曲/形变：螺母外缘或齿发生弯曲，可能导致装配干涉",
    "color": "颜色异常：表面变色、氧化或镀层不均，影响外观与防腐",
    "flip": "翻转/姿态错误：螺母被反向放置，属于装配姿态类缺陷",
    "scratch": "划痕：表面存在线性刮擦，影响外观与防腐能力",
    "good": "无缺陷",
}

SEVERITY_ORDER = {"Trace": 0, "Minor": 1, "Major": 2, "None": -1}


@dataclass
class Thresholds:
    image_score: float
    pixel_score: float
    defect_ratio: float
    source: str = "config"     # config | calibrated | manual


def resolve_thresholds(cfg: AppConfig, model: Optional[PatchCore],
                       overrides: Optional[Dict[str, float]] = None) -> Thresholds:
    """阈值优先级：命令行/人工覆盖 > 训练集自动标定 > 配置兜底。"""
    ov = overrides or {}
    cal = dict(getattr(model, "_calibration", None) or {}) if model is not None else {}
    if ov.get("image_score_threshold") is not None:
        image_score, source = float(ov["image_score_threshold"]), "manual"
    elif cal.get("image_score"):
        image_score, source = float(cal["image_score"]), "calibrated"
    else:
        image_score, source = cfg.report.image_score_threshold, "config"
    if ov.get("pixel_threshold") is not None:
        pixel = float(ov["pixel_threshold"])
    elif cal.get("pixel_score"):
        pixel = float(cal["pixel_score"])
    else:
        pixel = image_score * 0.95
    ratio = float(ov["defect_ratio_threshold"]) if ov.get("defect_ratio_threshold") is not None \
        else cfg.report.defect_ratio_threshold
    return Thresholds(image_score=float(image_score), pixel_score=float(pixel),
                      defect_ratio=float(ratio), source=source)


class DefectInspector:
    """对外唯一的推理入口：智能体的检测工具、评估脚本、报告插图都调用它。"""

    def __init__(self, cfg: AppConfig, model: Optional[PatchCore] = None,
                 thresholds: Optional[Thresholds] = None, overrides: Optional[Dict[str, float]] = None):
        self.cfg = cfg
        self.model = model or self._load_model()
        self.thresholds = thresholds or resolve_thresholds(cfg, self.model, overrides)
        self._train_items: Optional[List[ImgItem]] = None
        self._group_cache: Optional[Dict[str, List[ImgItem]]] = None

    # ---------------------------------------------------------- 初始化
    def _load_model(self) -> PatchCore:
        path = self.cfg.model_path()
        if not path.exists():
            raise FileNotFoundError(f"未找到模型 {path}，请先运行: python scripts/train.py")
        model = PatchCore.load(path, device=self.cfg.model.device)
        cal_path = Path(str(path) + ".calibration.json")
        if cal_path.exists():
            model._calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        return model

    @property
    def train_items(self) -> List[ImgItem]:
        if self._train_items is None:
            self._train_items = build_train_items(self.cfg)
        return self._train_items

    @property
    def reference_hot_score(self) -> float:
        """正常图热力图的典型上界（训练集 P95），用于计算缺陷区域的「局部显著度」。"""
        cal = getattr(self.model, "_calibration", {}) or {}
        ref = float(cal.get("hot_score_p95") or 0.0)
        return ref if ref > 0 else float(self.thresholds.pixel_score)

    def test_groups(self) -> Dict[str, List[ImgItem]]:
        if self._group_cache is None:
            self._group_cache = build_test_groups(self.cfg)
        return self._group_cache

    def calibrate(self, max_images: int = 120) -> Dict[str, float]:
        cal = self.model.calibrate(self.train_items[:max_images])
        self.thresholds = resolve_thresholds(self.cfg, self.model)
        return cal

    # ---------------------------------------------------------- 单图推理
    def _prepare(self, img: np.ndarray) -> np.ndarray:
        """把原图变换到与热力图相同的预处理几何（前景裁剪 + 缩放 + 居中）。"""
        t = preprocess_image(img, self.model.image_size, self.model.center_crop, self.model.auto_crop)
        return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def inspect_image(self, image_path: str | Path, out_dir: Optional[str | Path] = None,
                      save_visual: Optional[bool] = None) -> Dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        save_visual = bool(out_dir) if save_visual is None else save_visual
        t0 = time.time()
        img = load_image(image_path)
        tensor = preprocess_image(img, self.model.image_size, self.model.center_crop, self.model.auto_crop)
        scores, maps = self.model.predict_batch(tensor.unsqueeze(0))
        ms = (time.time() - t0) * 1000
        result = self._make_result(image_path, float(scores.item()), maps[0, 0].cpu().numpy())
        result["inference_ms"] = round(ms, 1)
        prepared = self._prepare(img)
        result["_prepared"] = prepared          # 供网页直接显示，无需再解码原图
        if save_visual:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            result["visuals"] = self._save_visuals(image_path.stem, prepared, result, out_dir)
        return result

    def _make_result(self, path: Path, score: float, amap: np.ndarray) -> Dict:
        """分数 + 原始热力图 -> 结构化结果（_map_norm 为内部字段，序列化时剔除）。

        热力图是 patch 网格高斯平滑后的渐变场，整幅图的动态范围会被压缩（如果像
        判级阈值那样用固定比例，缺陷边缘都会落在阈值以下）。因此：
        1. 先把热力图归一化到 [0,1]（1%~99% 分位截断，避免单个尖峰压暗整体）；
        2. 在归一化空间上按 `report.region_core_percentile` 取核心阈值提取缺陷区域；
        3. 缺陷的面积、位置、显著度全部基于该区域统计。
        判级（OK/NG）仍然只由图像级异常分数与训练集标定阈值决定，不受这里影响。
        """
        norm = self._normalize(amap)
        cfg_rep = self.cfg.report
        region_map = cv2.resize(norm, (self.cfg.model.center_crop, self.cfg.model.center_crop),
                                interpolation=cv2.INTER_CUBIC)
        core = float(np.clip(np.percentile(region_map, cfg_rep.region_core_percentile), 0.05, 0.97))
        binary, regions = postprocess(region_map, core)
        # 归一化值还原回原始热力图分数尺度，报告里的峰值与阈值可直接比较
        lo_n, span_n = self._norm_bounds(amap)
        for r in regions:
            r["max_score"] = float(r["max_score"] * span_n + lo_n)
            r["mean_score"] = float(r["mean_score"] * span_n + lo_n)
        top = max((r["max_score"] for r in regions), default=float(amap.max()))
        for r in regions:
            # 核心显著性：该区域峰值相对本图最异常点的比例（0~1），衡量「是不是主角」
            r["significance"] = float(r["max_score"] / top) if top > 0 else 0.0
            r["severity"] = describe_severity(r, cfg_rep)
        focus = [r for r in regions if r["significance"] >= cfg_rep.local_significance]
        total_ratio = float(sum(r["area_ratio"] for r in focus))
        reason = []
        if score >= self.thresholds.image_score:
            reason.append("异常分数超阈值")
        if focus and total_ratio >= cfg_rep.min_area_ratio:
            reason.append("存在局部显著异常区域")
        verdict = "OK" if (score < self.thresholds.image_score or not focus or
                           total_ratio < cfg_rep.defect_ratio_threshold) else "NG"
        severity = "None"
        if verdict == "NG":
            severity = max((r["severity"] for r in focus), key=lambda s: SEVERITY_ORDER.get(s, 0))
        return {
            "image_name": path.name,
            "image_path": str(path),
            "verdict": verdict,
            "verdict_reason": "；".join(reason) if verdict == "NG" else
            ("异常分数未超过阈值" if score < self.thresholds.image_score else "无局部显著异常区域"),
            "anomaly_score": round(score, 5),
            "threshold": round(self.thresholds.image_score, 5),
            "score_margin": round(score - self.thresholds.image_score, 5),
            "threshold_source": self.thresholds.source,
            "pixel_threshold": round(self.thresholds.pixel_score, 5),
            "ratio_vs_threshold": round(score / max(1e-8, self.thresholds.image_score), 4),
            "max_significance": round(max((r["significance"] for r in regions), default=0.0), 3),
            "defect_ratio": round(total_ratio, 6),
            "defect_ratio_threshold": cfg_rep.min_area_ratio,
            "num_regions": len(focus),
            "regions": [_fmt_region(r) for r in focus] if verdict == "NG" else [],
            "severity": severity,
            "recommendation": recommendation_for(verdict, severity),
            "model": {"name": "PatchCore", "backbone": self.model.backbone,
                      "image_size": self.model.image_size, "auto_crop": self.model.auto_crop,
                      "coreset": int(self.model.memory.features.shape[0]) if self.model.memory else 0},
            "_map_norm": norm,
        }

    @staticmethod
    def _normalize(amap: np.ndarray) -> np.ndarray:
        """热力图归一化到 [0,1]，并用 99 分位数截断，避免单个尖峰把其余区域压黑。"""
        a = np.asarray(amap, dtype=np.float32)
        lo = float(np.percentile(a, 1))
        hi = float(np.percentile(a, 99))
        if hi <= lo + 1e-6:
            hi = float(a.max())
            lo = float(a.min())
        return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    @staticmethod
    def _norm_bounds(amap: np.ndarray) -> tuple:
        """与 _normalize 完全一致的 (lo, hi)，用于把归一化值还原为原始分数。"""
        a = np.asarray(amap, dtype=np.float32)
        lo = float(np.percentile(a, 1))
        hi = float(np.percentile(a, 99))
        if hi <= lo + 1e-6:
            hi = float(a.max())
            lo = float(a.min())
        return lo, hi - lo + 1e-8

    def _save_visuals(self, stem: str, img: np.ndarray, result: Dict, out_dir: Path) -> Dict[str, str]:
        amap = result["_map_norm"]
        h, w = img.shape[:2]
        regions_px = []
        for r in result["regions"]:
            x0, y0, x1, y1 = r["bbox_norm"]
            regions_px.append(dict(r, bbox_px=[int(x0 * w), int(y0 * h),
                                               max(1, int((x1 - x0) * w)), max(1, int((y1 - y0) * h))]))
        # 区域掩码：把报告里的真实缺陷轮廓画到图上（而不是热力图整片）
        pix = cv2.resize(amap, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in regions_px:
            x, y, rw, rh = r["bbox_px"]
            sub = pix[y:y + rh, x:x + rw]
            if sub.size:
                thr = float(np.percentile(sub, 70))
                mask[y:y + rh, x:x + rw] |= (sub >= thr).astype(np.uint8)
        overlay = mask_overlay_png(img, amap, regions_px, mask=mask)
        # 矢量素材和叠加图都放到报告目录下的 assets 子目录，报告可以直接拷走
        svg_dir = out_dir / "assets"
        svg_dir.mkdir(parents=True, exist_ok=True)
        overlay_png = svg_dir / f"{stem}_overlay.png"
        cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tofile(str(overlay_png))
        heat_svg = svg_dir / f"{stem}_heatmap.svg"
        heat_svg.write_text(heatmap_overlay_svg(img, amap, regions_px, mask=mask), encoding="utf-8")
        hist_svg = svg_dir / f"{stem}_hist.svg"
        hist_svg.write_text(score_histogram_svg(amap, self.thresholds.pixel_score), encoding="utf-8")
        return {"overlay_png": str(overlay_png), "heatmap_svg": str(heat_svg), "hist_svg": str(hist_svg)}

    # ---------------------------------------------------------- 批量推理
    def inspect_many(self, paths: List[str | Path], out_dir: Optional[str | Path] = None,
                     max_visuals: int = 6) -> List[Dict]:
        items: List[ImgItem] = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"图片不存在: {path}")
            items.append(ImgItem(path, -1, "unknown"))
        score_chunks, map_chunks = [], []
        for batch, _ in iter_batches(items, self.model.image_size, self.model.center_crop,
                                     self.model.batch_size, self.model.auto_crop):
            s, m = self.model.predict_batch(batch)
            score_chunks.append(s.cpu().numpy())
            map_chunks.append(m.cpu().numpy())
        scores = np.concatenate(score_chunks)
        maps = np.concatenate(map_chunks)
        results: List[Dict] = []
        for i, (it, sc, am) in enumerate(zip(items, scores, maps)):
            r = self._make_result(it.path, float(sc), am[0])
            if out_dir is not None and i < max_visuals:
                prepared = self._prepare(load_image(it.path))
                r["visuals"] = self._save_visuals(it.path.stem, prepared, r, Path(out_dir))
            results.append(r)
        return results

    def inspect_folder(self, folder: str | Path, limit: Optional[int] = None,
                       out_dir: Optional[str | Path] = None, max_visuals: int = 6) -> List[Dict]:
        files = list_images(Path(folder))
        if limit:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"目录中没有图片: {folder}")
        return self.inspect_many(files, out_dir=out_dir, max_visuals=max_visuals)


# ---------------------------------------------------------------- 辅助
def recommendation_for(verdict: str, severity: str) -> str:
    if verdict == "OK":
        return "放行：未检出不超出阈值的异常区域"
    return {
        "Major": "拦截：转人工复检，评估返修或报废",
        "Minor": "隔离：人工抽检确认，建议跟踪为趋势项",
        "Trace": "提示：标记并统计，可按工艺规程放行",
    }.get(severity, "隔离：转人工复检")


def _fmt_region(r: Dict) -> Dict:
    return {
        "bbox_norm": [round(float(v), 4) for v in r["bbox"]],
        "bbox_px": [int(v) for v in r["bbox_px"]],
        "area_ratio": round(float(r["area_ratio"]), 6),
        "mean_score": round(float(r["mean_score"]), 4),
        "max_score": round(float(r["max_score"]), 4),
        "significance": round(float(r.get("significance", 0.0)), 3),
        "location": r["centroid_zone"],
        "location_detail": describe_location(r["bbox"]),
        "severity": r["severity"],
    }


def describe_location(bbox, zone_enabled: bool = False) -> str:
    """把归一化 bbox 转成人类可读的位置描述（按图像宽高的一半划分左右/上下）。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    horiz = "左半部" if cx < 0.5 else "右半部"
    vert = "上半部" if cy < 0.5 else "下半部"
    if zone_enabled:
        return zone_of(cx, cy)
    return f"{horiz} - {vert}"


def strip_internal(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def jsonable(result: Dict) -> Dict:
    return json.loads(json.dumps(strip_internal(result), ensure_ascii=False, default=float))
