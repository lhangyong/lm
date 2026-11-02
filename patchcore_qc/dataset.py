"""MVTec-AD 风格数据集读取与预处理。

目录结构（本机数据 E:\\Projects\\shixun\\jslm\\metal_nut）::

    metal_nut/
        train/good/*.png                 无缺陷训练图
        test/{good,bent,color,flip,scratch}/*.png
        ground_truth/{bent,color,flip,scratch}/*_mask.png   像素级标注

label 约定：0 = good（正常），1 = 缺陷。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    """按需读取图片并做「缩放 + 中心裁剪 + ToTensor」预处理。"""

    def __init__(self, items: Sequence["ImgItem"], image_size: int = 256,
                 center_crop: Optional[int] = 256):
        self.items = list(items)
        self.image_size = image_size
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.items)

    @property
    def masks(self) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.items[i].mask for i in range(len(self.items))]))
    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = load_image(item.path)
        img = preprocess_image(img, self.image_size, self.center_crop)
        return img, idx


@dataclass
class ImgItem:
    path: Path
    label: int
    defect_type: str
    mask: Optional[np.ndarray] = None      # HxW float32，0/1
    mask_path: Optional[Path] = None


# ---------------------------------------------------------------- 基础工具
def load_image(path: str | Path) -> np.ndarray:
    """读图：BGR uint8 -> RGB uint8（用 imdecode 兼容中文路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码: {path}")
    return (mask > 0).astype(np.float32)


def preprocess_image(img: np.ndarray, image_size: int = 256,
                     center_crop: Optional[int] = 256, auto_crop: bool = False) -> torch.Tensor:
    """图片 -> 归一化 tensor（缩放短边 + 中心裁剪）。

    auto_crop=True 时（金属/圆形零件推荐），先按「前景包围盒」裁成正方形再缩放：
    螺母这类零件的关键差异都在主体上，几何对齐后 patch 更容易比对。
    """
    img = crop_to_foreground(img) if auto_crop else img
    h, w = img.shape[:2]
    scale = image_size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    ch, cw = center_crop or image_size, center_crop or image_size
    top, left = max(0, (nh - ch) // 2), max(0, (nw - cw) // 2)
    img = img[top:top + ch, left:left + cw]
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1))


def _median_border(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    bw = max(2, min(h, w) // 32)
    border = np.concatenate([img[:bw].reshape(-1, 3), img[-bw:].reshape(-1, 3),
                             img[:, :bw].reshape(-1, 3), img[:, -bw:].reshape(-1, 3)])
    return np.median(border, axis=0)


def foreground_mask(img: np.ndarray, thresh: int = 22) -> np.ndarray:
    """估计前景（零件）区域：与图像边缘背景差异超过阈值的像素 + 形态学清理。"""
    med = _median_border(img)
    diff = np.abs(img.astype(np.int16) - med.astype(np.int16)).max(axis=2)
    mask = (diff > thresh).astype(np.uint8)
    k = max(3, (min(img.shape[:2]) // 16) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.ones(img.shape[:2], np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.02 * img.shape[0] * img.shape[1]:
        return np.ones(img.shape[:2], np.uint8)     # 前景太小，认为是整幅图
    return (labels == largest).astype(np.uint8)


def foreground_crop_box(img: np.ndarray, margin: float = 0.06,
                        min_scale: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
    """计算前景正方形裁剪框 (y0, x0, y1, x1)；前景过小或无前景时返回 None。"""
    mask = foreground_mask(img)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    h, w = img.shape[:2]
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0)
    if side < min_scale * min(h, w):
        return None
    side = side + 2 * int(round(side * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, cy - side // 2 + side, cx - side // 2 + side


def crop_to_foreground(img: np.ndarray, margin: float = 0.06, min_scale: float = 0.25) -> np.ndarray:
    """按前景包围盒裁成正方形（含 margin）；裁剪过猛时回退为原图。"""
    box = foreground_crop_box(img, margin, min_scale)
    if box is None:
        return img
    h, w = img.shape[:2]
    y0, x0, y1, x1 = box
    pad_t, pad_l = max(0, -y0), max(0, -x0)
    pad_b, pad_r = max(0, y1 - h), max(0, x1 - w)
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pad_t or pad_l or pad_b or pad_r:
        bg = _median_border(img).astype(np.uint8)
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                  value=tuple(int(v) for v in bg))
    return np.ascontiguousarray(crop)


def preprocess_mask(mask: np.ndarray, image_size: int = 256,
                    center_crop: Optional[int] = 256, auto_crop: bool = False,
                    ref_img: Optional[np.ndarray] = None) -> np.ndarray:
    """掩码与图片使用完全一致的几何变换（区域用最近邻，保证 0/1）。"""
    if auto_crop and ref_img is not None:
        box = foreground_crop_box(ref_img)
        if box is not None:
            h, w = mask.shape[:2]
            y0, x0, y1, x1 = box
            out = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=mask.dtype)
            sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(h, y1), min(w, x1)
            if sy1 > sy0 and sx1 > sx0:
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1]
            mask = out
    h, w = mask.shape[:2]
    scale = image_size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ch, cw = center_crop or image_size, center_crop or image_size
    top, left = max(0, (nh - ch) // 2), max(0, (nw - cw) // 2)
    mask = mask[top:top + ch, left:left + cw]
    return (mask > 0.5).astype(np.float32)


# ---------------------------------------------------------------- 数据集构建
def list_images(folder: Path) -> List[Path]:
    if not Path(folder).exists():
        return []
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def build_train_items(cfg) -> List[ImgItem]:
    """训练集只遍历 train/good（无监督：完全不看缺陷样本）。"""
    files = list_images(Path(cfg.train_dir))
    if not files:
        raise FileNotFoundError(f"训练集为空: {cfg.train_dir}")
    return [ImgItem(p, 0, "good") for p in files]


def build_test_items(cfg, auto_crop: Optional[bool] = None) -> List[ImgItem]:
    auto_crop = bool(getattr(cfg.model, "auto_crop", False)) if auto_crop is None else auto_crop
    test_dir = Path(cfg.test_dir)
    gt_root = Path(cfg.gt_root)
    labels = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    items: List[ImgItem] = []
    for label in labels:
        files = list_images(test_dir / label)
        if not files:
            continue
        is_good = label == cfg.data.good_label
        if not is_good and gt_root.exists() and not (gt_root / label).exists():
            print(f"[warn] 缺陷类别 {label} 没有 ground_truth 目录，像素指标将跳过该类")
        for f in files:
            mask_path = None
            if not is_good:
                cand = gt_root / label / f"{f.stem}_mask.png"
                mask_path = cand if cand.exists() else None
            mask = None
            if mask_path is not None:
                raw = read_mask(mask_path)
                ref = load_image(f) if auto_crop else None
                mask = preprocess_mask(raw, cfg.model.image_size, cfg.model.center_crop,
                                       auto_crop=auto_crop, ref_img=ref)
            items.append(ImgItem(f, 0 if is_good else 1, label, mask, mask_path))
    if not items:
        raise FileNotFoundError(f"测试集为空: {cfg.test_dir}")
    return items


def build_test_groups(cfg) -> Dict[str, List[ImgItem]]:
    """按缺陷类别分组的测试样本：{'good': [...], 'bent': [...], ...}。"""
    groups: Dict[str, List[ImgItem]] = {}
    for it in build_test_items(cfg):
        groups.setdefault(it.defect_type, []).append(it)
    return groups


def balance_subset(items: Sequence[ImgItem], limit: int, seed: int = 0) -> List[ImgItem]:
    """按类别等比例抽样（用于快速验证流程）。"""
    if limit >= len(items):
        return list(items)
    rng = random.Random(seed)
    by_cls: dict[str, List[ImgItem]] = {}
    for it in items:
        by_cls.setdefault(it.defect_type, []).append(it)
    per = max(1, limit // len(by_cls))
    out: List[ImgItem] = []
    for cls in sorted(by_cls):
        out.extend(rng.sample(by_cls[cls], min(per, len(by_cls[cls]))))
    return out


def to_loader(ds: Dataset, batch_size: int = 8, num_workers: int = 0,
              shuffle: bool = False) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def iter_batches(items: Sequence[ImgItem], image_size: int, center_crop: Optional[int],
                 batch_size: int, auto_crop: bool = False) -> Iterator[Tuple[torch.Tensor, List[ImgItem]]]:
    """简易批处理迭代器（推理/调参用）。"""
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs = [preprocess_image(load_image(it.path), image_size, center_crop, auto_crop=auto_crop)
                for it in chunk]
        yield torch.stack(imgs), list(chunk)
