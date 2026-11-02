"""PatchCore 核心算法：记忆库构建、贪心 coreset 采样、kNN 打分与图像级聚合。

论文：Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.

流程：
  1. 训练集（仅正常样本）逐图提取 patch 特征 → （可选逐维标准化）→ 拼成 memory bank；
  2. 用贪心 k-center 从 memory bank 选出信息量子集（coreset，论文 0.1%~1%）；
  3. 推理时，每个测试 patch 取到 coreset 的最近邻距离；
  4. 用 coreset 自身 kNN 距离做重加权（越「孤僻」的正常 patch 权重越高）；
  5. 在 patch 网格上做高斯平滑得到热力图，取最大值作为图像级异常分数。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .dataset import ImgItem, iter_batches
from .feature_extractor import PatchEmbedding

FPS_PROJECTION_THRESHOLD = 20000   # 超过该规模先做随机子空间投影
FPS_PROJECTION_DIM = 256


# ================================================================ 基础算子
def compute_euclidean_distance(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    """成对欧氏距离，返回 [len(x), len(y)]。"""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    n_x, n_y = x.shape[0], y.shape[0]
    dist = x.pow(2).sum(-1).unsqueeze(-1).expand(n_x, n_y) \
        + y.pow(2).sum(-1).unsqueeze(0).expand(n_x, n_y) - 2 * x @ y.t()
    return dist.clamp_min_(eps).sqrt_()


def normalization_scale(features: np.ndarray) -> np.ndarray:
    """逐维标准化（官方实现默认开启；本数据集实测关闭后更好，可通过配置切换）。"""
    features = np.ascontiguousarray(features.astype(np.float32))
    features = features - features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-5] = 1.0
    return (features / std).astype(np.float32)


def farthest_point_sampling(embeddings: Tensor, sampling_ratio: float,
                            device: Optional[str] = None,
                            threshold: int = FPS_PROJECTION_THRESHOLD) -> Tensor:
    """贪心 k-center（最远点）采样，返回选中 patch 的下标 [n_sample]。

    随机初始点 + 每次选「离已选集合最远」的点；重复选中时加 1e-6 扰动跳出。
    N 很大时按论文 §4.1，先用随机子集 S 构造子空间基，把距离计算降到 O(N·|S|)。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = int(embeddings.shape[0])
    n_sample = max(1, min(n, int(round(sampling_ratio * n))))
    emb = embeddings.detach().float().to(device)
    if n > threshold:
        sub = torch.randperm(n, device=device)[:FPS_PROJECTION_DIM]
        basis = emb[sub].t()                                   # [C, |S|]
        # 分块投影，避免 N×|S| 一次性占用显存
        chunk = max(1, 4_000_000 // FPS_PROJECTION_DIM)
        parts = []
        for s in range(0, n, chunk):
            y = emb[s:s + chunk] @ basis
            parts.append(y / y.norm(dim=1, keepdim=True).clamp_min(1e-8))
        del basis
        feats = torch.cat(parts, dim=0)
    else:
        feats = emb

    dist_mat = torch.full((n,), float("inf"), device=device)
    chosen = [int(torch.randint(n, (1,), device=device).item())]
    for _ in range(n_sample - 1):
        d = compute_euclidean_distance(feats[chosen[-1]].unsqueeze(0), feats).squeeze(0)
        d = d + torch.rand_like(d) * 1e-6
        dist_mat = torch.minimum(dist_mat, d)
        chosen.append(int(torch.argmax(dist_mat).item()))
    return torch.tensor(chosen, dtype=torch.long)


# ================================================================ 记忆库
@dataclass
class MemoryBank:
    features: np.ndarray           # [M, C] coreset patch 特征
    patch_scores: np.ndarray       # [M] core patch 的 kNN 距离（重加权用）
    grid_size: int                 # patch 网格边长
    feature_dim: int
    meta: Dict = field(default_factory=dict)

    def save(self, path) -> None:
        import json
        np.savez_compressed(str(path), features=self.features, patch_scores=self.patch_scores,
                            grid_size=self.grid_size, feature_dim=self.feature_dim)
        Path(str(path) + ".json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    @classmethod
    def load(cls, path) -> "MemoryBank":
        import json
        z = np.load(str(path))
        meta = {}
        p = Path(str(path) + ".json")
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
        return cls(z["features"], z["patch_scores"], int(z["grid_size"]), int(z["feature_dim"]), meta)


# ================================================================ 检测器
class PatchCore:
    def __init__(self, backbone: str = "resnet18", layers: Sequence[str] = ("layer2", "layer3"),
                 image_size: int = 256, center_crop: Optional[int] = 256,
                 coreset_ratio: float = 0.01, max_coreset: int = 6000, n_neighbors: int = 1,
                 gaussian_sigma: float = 4.0, device: str = "auto", pretrained: bool = True,
                 batch_size: int = 8, num_workers: int = 0, auto_crop: bool = False,
                 local_pool: bool = True, normalize: bool = True):
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else \
            ("cpu" if device == "auto" else device)
        self.patch_embed = PatchEmbedding(backbone, layers, pretrained, local_pool).to(self.device).eval()
        self.backbone = backbone
        self.layers = list(layers)
        self.image_size = image_size
        self.center_crop = center_crop if center_crop is not None else image_size
        self.auto_crop = auto_crop
        self.local_pool = local_pool
        self.normalize = normalize
        self.coreset_ratio = coreset_ratio
        self.max_coreset = max_coreset
        self.n_neighbors = n_neighbors
        self.gaussian_sigma = gaussian_sigma
        self.feature_dim = self.patch_embed.feature_dim
        self.grid_size = self.patch_embed.grid_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.memory: Optional[MemoryBank] = None
        self.train_time: float = 0.0
        self._calibration: Dict[str, float] = {}
        self._core_t: Optional[Tensor] = None
        self._weights_t: Optional[Tensor] = None

    # ---------------------------------------------------------- 配置
    def config(self) -> Dict:
        return {"backbone": self.backbone, "layers": self.layers, "image_size": self.image_size,
                "center_crop": self.center_crop, "auto_crop": self.auto_crop,
                "local_pool": self.local_pool, "normalize": self.normalize,
                "coreset_ratio": self.coreset_ratio, "max_coreset": self.max_coreset,
                "n_neighbors": self.n_neighbors, "gaussian_sigma": self.gaussian_sigma,
                "feature_dim": self.feature_dim, "grid_size": self.grid_size,
                "batch_size": self.batch_size}

    @classmethod
    def from_config(cls, cfg, device: Optional[str] = None, pretrained: bool = True) -> "PatchCore":
        m = cfg.model
        return cls(backbone=m.backbone, layers=m.layers, image_size=m.image_size,
                   center_crop=m.center_crop, coreset_ratio=m.coreset_ratio,
                   max_coreset=m.max_coreset, n_neighbors=m.n_neighbors,
                   gaussian_sigma=m.gaussian_sigma, device=device or m.device,
                   pretrained=pretrained, batch_size=m.batch_size, num_workers=m.num_workers,
                   auto_crop=m.auto_crop, local_pool=m.local_pool, normalize=m.normalize)

    # ---------------------------------------------------------- 特征
    @torch.no_grad()
    def embed(self, images: Tensor) -> Tensor:
        feats = self.patch_embed(images.to(self.device)).float()
        side = int(round(feats.shape[1] ** 0.5))
        if self.memory is None and side * side == feats.shape[1]:
            self.grid_size = side
        return feats

    @torch.no_grad()
    def _embed_items(self, items: Sequence[ImgItem]) -> Tensor:
        feats = [self.embed(batch).cpu()
                 for batch, _ in iter_batches(items, self.image_size, self.center_crop,
                                              self.batch_size, self.auto_crop)]
        if not feats:
            raise ValueError("没有可推理的图片")
        return torch.cat(feats, dim=0)

    # ---------------------------------------------------------- 训练（构建记忆库）
    def fit(self, train_items: Sequence[ImgItem], verbose: bool = True) -> "PatchCore":
        t0 = time.time()
        all_feats = self._embed_items(train_items)
        n_img, _, c = all_feats.shape
        flat = all_feats.reshape(-1, c).numpy()
        if self.normalize:
            flat = normalization_scale(flat)
        if verbose:
            print(f"[PatchCore] 训练图片 {n_img} 张，patch 特征 {flat.shape[0]} x {c}"
                  f"（网格 {self.grid_size}x{self.grid_size}）")

        n_total = flat.shape[0]
        target = max(1, min(n_total, self.max_coreset, int(round(self.coreset_ratio * n_total))))
        corpus = torch.from_numpy(flat)
        if target < n_total:
            if verbose:
                print(f"[PatchCore] 贪心 coreset 采样 {n_total} → {target} ...")
            corpus = corpus[farthest_point_sampling(corpus, target / n_total, device=self.device).cpu()]
        coreset = corpus.cpu().numpy()

        # 重加权：coreset 中每个 patch 与最近 3 个 core patch 的平均距离
        core_t = torch.from_numpy(coreset).to(self.device)
        k = min(3, max(1, core_t.shape[0] - 1))
        w = np.zeros(core_t.shape[0], dtype=np.float32)
        for s in range(0, core_t.shape[0], 2048):
            e = min(s + 2048, core_t.shape[0])
            d = compute_euclidean_distance(core_t[s:e], core_t)
            rows = torch.arange(e - s, device=self.device)
            d[rows, rows + s] = float("inf")
            w[s:e] = torch.topk(d, k=k, dim=1, largest=False).values.mean(dim=1).cpu().numpy()
        w_max = float(w.max()) if w.size and w.max() > 0 else 1.0

        self.memory = MemoryBank(coreset, w / w_max, self.grid_size, c, meta={})
        self.memory.meta = {**self.config(), "n_train_images": int(n_img), "n_train_patches": int(n_total),
                            "n_coreset": int(coreset.shape[0]), "train_time_s": 0.0}
        self._refresh_cache()
        self.train_time = time.time() - t0
        self.memory.meta["train_time_s"] = round(self.train_time, 2)
        if verbose:
            print(f"[PatchCore] 记忆库构建完成：{coreset.shape[0]} 个 core patch，耗时 {self.train_time:.1f}s")
        return self

    def _refresh_cache(self) -> None:
        assert self.memory is not None
        w = self.memory.patch_scores.astype(np.float32)
        w = w / (w.max() + 1e-8)
        self.grid_size = self.memory.grid_size
        self._core_t = torch.from_numpy(self.memory.features).to(self.device)
        self._weights_t = torch.from_numpy(w).to(self.device)

    # ---------------------------------------------------------- 打分
    @torch.no_grad()
    def patch_distances(self, embeddings: Tensor) -> Tensor:
        """[B,N,C] → [B,N]：每个 patch 到 coreset 的（重加权）最近邻距离。"""
        assert self._core_t is not None, "请先 fit() 或 PatchCore.load()"
        b, n, c = embeddings.shape
        q = embeddings.reshape(-1, c)
        m = self._core_t.shape[0]
        out = torch.empty(q.shape[0], device=q.device)
        step = max(1, 131072 // max(1, m))
        k = min(self.n_neighbors, m)
        for s in range(0, q.shape[0], step):
            d = compute_euclidean_distance(q[s:s + step], self._core_t)
            nn = torch.topk(d, k=k, dim=1, largest=False)
            if k == 1:
                out[s:s + step] = nn.values[:, 0]
            else:
                ww = self._weights_t[nn.indices]
                out[s:s + step] = (nn.values * ww).sum(dim=1) / (ww.sum(dim=1) + 1e-8)
        return out.reshape(b, n)

    def score_map_from_patches(self, patch_scores: Tensor) -> Tensor:
        b = patch_scores.shape[0]
        smap = patch_scores.reshape(b, 1, self.grid_size, self.grid_size)
        return gaussian_blur(smap, sigma=self.gaussian_sigma)

    @torch.no_grad()
    def predict_batch(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """返回 (图像级分数 [B], 热力图 [B,h,w])。"""
        amap = self.score_map_from_patches(self.patch_distances(self.embed(images)))
        return amap.flatten(1).max(dim=1).values, amap

    @torch.no_grad()
    def predict_items(self, items: Sequence[ImgItem], verbose: bool = True) -> Dict[str, np.ndarray]:
        scores: List[np.ndarray] = []
        maps: List[np.ndarray] = []
        total = len(items)
        for i, (batch, _) in enumerate(iter_batches(items, self.image_size, self.center_crop,
                                                    self.batch_size, self.auto_crop)):
            s, m = self.predict_batch(batch)
            scores.append(s.cpu().numpy())
            maps.append(m.cpu().numpy())
            if verbose and ((i + 1) % 20 == 0 or (i + 1) * self.batch_size >= total):
                print(f"  推理 {min(total, (i + 1) * self.batch_size)}/{total}")
        return {"scores": np.concatenate(scores), "maps": np.concatenate(maps)}

    # ---------------------------------------------------------- 标定
    def calibrate(self, train_items: Sequence[ImgItem]) -> Dict[str, float]:
        """用正常训练样本标定阈值：图像阈值 = max(P99.5, mean+3σ)；像素阈值 = 热力图 P99.9。"""
        res = self.predict_items(train_items, verbose=False)
        s, maps = res["scores"], res["maps"]
        cal = {
            "image_score": float(max(np.percentile(s, 99.5), s.mean() + 3 * s.std())),
            "pixel_score": float(max(np.percentile(maps, 99.9), float(maps.max()) * 0.5)),
            # 正常图热力图的典型上界，报告里「局部显著度」以此为参照
            "hot_score_p95": float(np.percentile(maps, 95)),
            "train_mean": float(s.mean()), "train_std": float(s.std()),
            "train_p99_5": float(np.percentile(s, 99.5)), "train_max": float(s.max()),
            "n_images": int(len(s)),
        }
        self._calibration = cal
        return cal

    # ---------------------------------------------------------- 持久化
    def save(self, path) -> None:
        assert self.memory is not None, "没有可保存的记忆库"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"patchcore": self.config(),
                    "memory": {"features": torch.from_numpy(self.memory.features),
                               "patch_scores": torch.from_numpy(self.memory.patch_scores),
                               "meta": self.memory.meta}},
                   str(path))

    @classmethod
    def load(cls, path, device: str = "auto") -> "PatchCore":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        p = dict(ckpt["patchcore"])
        model = cls(backbone=p["backbone"], layers=p["layers"], image_size=p["image_size"],
                    center_crop=p["center_crop"], coreset_ratio=p["coreset_ratio"],
                    max_coreset=p["max_coreset"], n_neighbors=p["n_neighbors"],
                    gaussian_sigma=p["gaussian_sigma"], device=device, pretrained=True,
                    auto_crop=p.get("auto_crop", False), local_pool=p.get("local_pool", True),
                    normalize=p.get("normalize", True))
        mem = ckpt["memory"]
        gs = int(mem["meta"].get("grid_size", p.get("grid_size", model.grid_size)))
        model.memory = MemoryBank(mem["features"].numpy().astype(np.float32),
                                  mem["patch_scores"].numpy().astype(np.float32),
                                  gs, int(p.get("feature_dim", model.feature_dim)), mem.get("meta", {}))
        model._refresh_cache()
        model.train_time = float(mem.get("meta", {}).get("train_time_s", 0.0))
        return model


# ================================================================ 后处理
def gaussian_blur(x: Tensor, sigma: float = 4.0) -> Tensor:
    """[B,1,H,W] 可分离高斯平滑。"""
    if not sigma or sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    x = F.conv2d(x, g.view(1, 1, -1, 1), padding=(radius, 0))
    x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, radius))
    return x


def postprocess(anomaly_map: np.ndarray, threshold: float,
                connectivity: int = 8, min_area: int = 4) -> Tuple[np.ndarray, List[Dict]]:
    """阈值二值化 + 连通域分析，返回掩码与缺陷区域（坐标为归一化值）。

    说明：只保留落在 **核心阈值区**（≥ threshold + 15% 余量）的连通域作为候选，
    再把该连通域在较低阈值下的部分纳入同一区域，这样既不会因单一高斯平滑把
    「略高于阈值的整片背景」算成一个巨大区域，也能保留缺陷的真实轮廓。
    """
    import cv2
    m = anomaly_map.astype(np.float32)
    base = m >= threshold                      # 轮廓阈值
    core = m >= threshold + 0.15 * (1.0 - threshold)   # 区域检测用的核心阈值
    n, labels, _, _ = cv2.connectedComponentsWithStats(base.astype(np.uint8), connectivity=connectivity)
    n_core, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8),
                                                                          connectivity=connectivity)
    keep = np.zeros(n_core, dtype=np.uint8)
    for i in range(1, n_core):
        if int(core_stats[i, cv2.CC_STAT_AREA]) >= 1:
            keep[i] = 1
    core_region = keep[core_labels]            # 每个像素所属核心连通域是否保留

    combined = np.where((labels > 0) & (core_region > 0), labels, 0).astype(np.int32)
    # 无核心的区域直接丢弃；再用 base 掩码还原轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS if connectivity == 4 else cv2.MORPH_RECT, (3, 3))
    binary = (combined > 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1) * base
    if binary.sum() == 0:                      # 全都不显著
        return binary, []

    n2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    h, w = anomaly_map.shape
    regions: List[Dict] = []
    for i in range(1, n2):
        area = int(stats2[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, rw, rh = (int(v) for v in stats2[i, :4])
        vals = m[y:y + rh, x:x + rw][labels2[y:y + rh, x:x + rw] == i]
        cx, cy = float(centroids2[i][0]) / w, float(centroids2[i][1]) / h
        regions.append({"bbox": [x / w, y / h, (x + rw) / w, (y + rh) / h],
                        "bbox_px": [x, y, rw, rh],
                        "area_ratio": area / float(h * w),
                        "mean_score": float(vals.mean()),
                        "max_score": float(vals.max()),
                        "centroid": [cx, cy],
                        "centroid_zone": zone_of(cx, cy)})
    regions.sort(key=lambda r: r["area_ratio"] * r["max_score"], reverse=True)
    return binary, regions


def zone_of(x: float, y: float) -> str:
    """归一化坐标 → 九宫格方位词，便于在报告里用人话描述位置。"""
    col = "左" if x < 1 / 3 else ("中" if x < 2 / 3 else "右")
    row = "上" if y < 1 / 3 else ("中" if y < 2 / 3 else "下")
    if col == "中" and row == "中":
        return "图像中心"
    if row == "中":
        return f"{col}侧中部"
    if col == "中":
        return f"中央{row}部"
    return f"{col}{row}角"


def describe_severity(region: Dict, cfg_report) -> str:
    """按异常面积占比分级：Major（显著）/ Minor（轻微）/ Trace（极微小）。"""
    ar = region["area_ratio"]
    if ar >= cfg_report.max_area_ratio:
        return "Major"
    if ar >= cfg_report.min_area_ratio:
        return "Minor"
    return "Trace"
