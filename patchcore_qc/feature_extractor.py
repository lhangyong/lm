"""骨干网络特征提取。

参考 PatchCore 官方实现（amazon-science/patchcore-inspection）：
- 取中间层 layer2 / layer3（ImageNet 预训练权重）拼接；
- 每个 patch 特征在当前 batch 内做 3x3 邻域平均池化（局部上下文聚合）。

特征尺寸：256x256 输入 -> resnet18 的 layer2 为 32x32 网格，layer3 上采样到 32x32,
每个 patch 维度 = 128 + 256 = 384；wide_resnet50_2 为 512 + 1024 = 1536。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, Wide_ResNet50_2_Weights, resnet18, wide_resnet50_2

_BACKBONES = {
    # backbone: (构造函数, 权重, patch 边长下采样倍数, 通道 dict)
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, 16,
                 {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2, 32,
                        {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}),
}


class FeatureExtractor(nn.Module):
    """返回 [B, C, h, w] 的多层拼接特征图。"""

    def __init__(self, backbone: str = "resnet18",
                 layers: Sequence[str] = ("layer2", "layer3"), pretrained: bool = True):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"不支持的 backbone: {backbone}，可选 {list(_BACKBONES)}")
        ctor, weights, self.long_side, channels = _BACKBONES[backbone]
        self.backbone = backbone
        self.layers = list(layers)
        self.out_channels: List[int] = [channels[name] for name in self.layers]
        self.feature_dim = sum(self.out_channels)

        net = ctor(weights=weights if pretrained else None)
        # 注意：不能直接 self.xxx = net.stem，否则参数不会随 .to(device) 迁移
        self.conv1, self.bn1, self.relu, self.maxpool = (
            net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleDict({
            name: getattr(net, name) for name in ("layer1", "layer2", "layer3", "layer4")
        })
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):  # 始终保持 eval：BN 统计量固定
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        outs: List[Tensor] = []
        h = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = self.stages[name](h)
            if name in self.layers:
                outs.append(h)
        target = outs[0].shape[-2:]
        aligned = []
        for i, feat in enumerate(outs):
            feat = F.interpolate(feat, size=target, mode="bilinear", align_corners=False)
            if i > 0:
                # 官方实现在邻域聚合前对深层特征做恒定值 padding，
                # 这里等价地用「边界复制」抹掉线性插值在边缘产生的伪影。
                feat = F.pad(feat, (1, 1, 1, 1), mode="replicate")[..., 1:-1, 1:-1]
            aligned.append(feat)
        return torch.cat(aligned, dim=1)


class PatchEmbedding(nn.Module):
    """骨干特征 -> [B, N, C] patch 描述子（含 3x3 邻域平均）。"""

    def __init__(self, backbone: str = "resnet18", layers: Sequence[str] = ("layer2", "layer3"),
                 pretrained: bool = True, local_pool: bool = True):
        super().__init__()
        self.extractor = FeatureExtractor(backbone, layers, pretrained)
        self.long_side = self.extractor.long_side
        self.feature_dim = self.extractor.feature_dim
        self.local_pool = local_pool
        self._grid_side = max(1, 256 // self.long_side)   # 首次前向时按实际输入更新
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    @property
    def grid_size(self) -> int:
        """patch 网格边长（256x256 输入时为 32）。"""
        return self._grid_side

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        feats = self.extractor(x)
        self._grid_side = int(feats.shape[-2])
        if self.local_pool:
            feats = self.avg(feats)
        return feats.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def feature_map(self, x: Tensor) -> Tensor:
        """未平滑的 [B, C, h, w]，用于可视化。"""
        return self.extractor(x)
