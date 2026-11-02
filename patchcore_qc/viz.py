"""可视化工具：不依赖 matplotlib，直接产出 PNG / SVG 素材。

- mask_overlay_png：原图 + 异常热力图（jet 色表）+ 缺陷框，用于人工复核；
- heatmap_overlay_svg：矢量版热力图（32x32 色块），可嵌入 HTML 报告；
- score_histogram_svg：异常分数直方图。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# jet 近似色表（0->蓝 1->红）
_X = np.linspace(0, 1, 256)
JET_LUT = (np.stack([np.clip(1.5 - np.abs(4 * _X - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * _X - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * _X - 1), 0, 1)], axis=1) * 255).astype(np.uint8)

SEVERITY_COLOR = {"Major": (255, 80, 40), "Minor": (255, 200, 0), "Trace": (0, 200, 255)}
SEVERITY_COLOR_HEX = {"Major": "#ff5030", "Minor": "#ffc800", "Trace": "#00c8ff"}


def colorize(amap_norm: np.ndarray) -> np.ndarray:
    """归一化热力图 [0,1] -> RGB uint8。"""
    idx = np.clip((np.asarray(amap_norm) * 255).astype(np.int32), 0, 255)
    return JET_LUT[idx]


def mask_overlay_png(image_rgb: np.ndarray, amap_norm: np.ndarray,
                     regions_px: Sequence[Dict], alpha: float = 0.45,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """原图叠加热力图；mask 给出时只在该区域着色并描出轮廓，同时用彩色框标出各缺陷。"""
    h, w = image_rgb.shape[:2]
    heat = cv2.resize(colorize(amap_norm), (w, h), interpolation=cv2.INTER_CUBIC)
    weight = cv2.resize(np.asarray(amap_norm, dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    weight = np.clip(weight * 1.4, 0, 1) * alpha
    if mask is not None and int(np.asarray(mask).sum()) > 0:
        keep = cv2.resize(np.asarray(mask, dtype=np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        weight = weight * (keep > 0)
    out = image_rgb.astype(np.float32) * (1 - weight[..., None]) + heat.astype(np.float32) * weight[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    thick = max(1, int(round(min(h, w) / 200)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.35, min(h, w) / 900)
    if mask is not None and int(np.asarray(mask).sum()) > 0:
        cnts, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), thick)
    for i, r in enumerate(regions_px[:12]):
        x, y, bw, bh = (int(v) for v in r["bbox_px"])
        col = SEVERITY_COLOR.get(r["severity"], (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, thick)
        if i < 6:
            label = f"#{i + 1} {r['severity']} {r['max_score']:.2f}"
            cv2.putText(out, label, (x + 2, max(12, y - 4)), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def heatmap_overlay_svg(image_rgb: np.ndarray, amap_norm: np.ndarray,
                        regions_px: Sequence[Dict], cell: int = 6,
                        mask: Optional[np.ndarray] = None) -> str:
    """用色块拼出热力图 SVG（网格 32x32 -> 约 1024 个 rect），可内嵌 HTML。

    mask 给出时只显示缺陷区域内的色块，其余区域压暗，突出定位结果。
    """
    gh, gw = np.asarray(amap_norm).shape[:2]
    W, H = gw * cell * 4, gh * cell * 4
    m = None
    if mask is not None and int(np.asarray(mask).sum()) > 0:
        m = cv2.resize(np.asarray(mask, dtype=np.float32), (gw, gh), interpolation=cv2.INTER_NEAREST)
    elems: List[str] = []
    for y in range(gh):
        for x in range(gw):
            v = float(np.clip(amap_norm[y, x], 0, 1))
            r, g, b = (int(c) for c in JET_LUT[int(v * 255)])
            opacity = 0.15 + 0.85 * v
            if m is not None and m[y, x] <= 0:
                opacity *= 0.18
            elems.append(f'<rect x="{x * cell * 4}" y="{y * cell * 4}" width="{cell * 4}" '
                         f'height="{cell * 4}" fill="rgb({r},{g},{b})" fill-opacity="{opacity:.2f}"/>')
    boxes: List[str] = []
    for i, r in enumerate(regions_px[:12]):
        x0, y0, x1, y1 = r["bbox_norm"]
        sx, sy, sw, sh = x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H
        col = SEVERITY_COLOR_HEX.get(r["severity"], "#00ff88")
        boxes.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" fill="none" '
                     f'stroke="{col}" stroke-width="3"/>')
        boxes.append(f'<text x="{sx + 4:.1f}" y="{min(H - 4, max(14, sy - 6)):.1f}" fill="{col}" '
                     f'font-size="13" font-family="monospace">#{i + 1} {r["severity"]}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" '
            f'style="background:#111;border-radius:8px" shape-rendering="crispEdges">'
            f'<rect width="{W}" height="{H}" fill="#111"/>' + "".join(elems) + "".join(boxes) + "</svg>")


def score_histogram_svg(amap_norm: np.ndarray, absolute_threshold: float,
                        bins: int = 48, width: int = 520, height: int = 190) -> str:
    """patch 异常分数直方图（横轴为绝对分数），虚线标注像素阈值。"""
    vals = np.asarray(amap_norm, dtype=np.float64).flatten()
    hi = max(float(absolute_threshold) * 1.03, float(np.percentile(vals, 99.9)), 1e-6)
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, hi))
    hist = hist / (hist.max() + 1e-8)
    pad_l, pad_b, pad_t, pad_r = 10, 22, 10, 10
    bw = (width - pad_l - pad_r) / bins
    bars = []
    for i, v in enumerate(hist):
        bh = v * (height - pad_t - pad_b)
        center = (edges[i] + edges[i + 1]) / 2
        col = "#ff6b6b" if center >= float(absolute_threshold) else "#4aa8ff"
        bars.append(f'<rect x="{pad_l + i * bw + 1:.1f}" y="{height - pad_b - bh:.1f}" '
                    f'width="{bw - 2:.1f}" height="{bh:.1f}" fill="{col}" rx="1"/>')
    tx = pad_l + float(np.clip(float(absolute_threshold) / hi, 0, 1)) * (width - pad_l - pad_r)
    thr = (f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{pad_t}" y2="{height - pad_b}" stroke="#ffd166" '
           f'stroke-dasharray="4 3" stroke-width="2"/>'
           f'<text x="{min(tx + 4, width - 110):.1f}" y="{pad_t + 12}" fill="#ffd166" font-size="11">'
           f'阈值 {float(absolute_threshold):.3f}</text>')
    labels = (f'<text x="{pad_l}" y="{height - 6}" fill="#888" font-size="11">0</text>'
              f'<text x="{width / 2 - 12:.0f}" y="{height - 6}" fill="#888" font-size="11">{hi / 2:.2f}</text>'
              f'<text x="{width - pad_r - 40}" y="{height - 6}" fill="#888" font-size="11">{hi:.2f}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
            f'style="background:#16181d;border-radius:8px">' + "".join(bars) + thr + labels + "</svg>")


def montage(overlay_paths: Sequence[str], max_items: int = 8, thumb: int = 240) -> Optional[np.ndarray]:
    """把若干张叠加图拼成一张网格图（BGR，用于报告封面）。"""
    imgs = []
    for p in overlay_paths[:max_items]:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(cv2.resize(img, (thumb, thumb)))
    if not imgs:
        return None
    cols = min(4, len(imgs))
    rows = int(np.ceil(len(imgs) / cols))
    canvas = np.full((rows * thumb, cols * thumb, 3), 30, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = im
    return canvas
