"""配置与入口：把路径、模型、阈值、智能体参数集中到一个 AppConfig 里。

优先级：命令行参数 > configs/default.yaml > 本文件默认值 > 环境变量。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("PATCHCORE_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
MODEL_DIR = OUTPUT_ROOT / "models"
REPORT_DIR = OUTPUT_ROOT / "reports"
RUN_DIR = OUTPUT_ROOT / "runs"
VIS_DIR = OUTPUT_ROOT / "visualizations"
DATA_ROOT = Path(os.environ.get("PATCHCORE_DATA_ROOT", r"E:\Projects\shixun\jslm"))


@dataclass
class DataConfig:
    root: str = str(DATA_ROOT)
    dataset: str = "metal_nut"
    train_split: str = "train/good"
    test_dir: str = "test"
    gt_dir: str = "ground_truth"
    good_label: str = "good"


@dataclass
class ModelConfig:
    """PatchCore 超参。默认配置为「不带 3x3 邻域平滑 + 前景裁剪」，见 README 的消融表。"""
    backbone: str = "resnet18"                 # resnet18 | wide_resnet50_2
    layers: List[str] = field(default_factory=lambda: ["layer2", "layer3"])
    image_size: int = 256                      # 缩放后的短边（auto_crop=True 时是裁剪后的边长）
    center_crop: int = 256
    auto_crop: bool = True                     # 先按前景包围盒裁剪成正方形
    local_pool: bool = False                   # 是否做 3x3 邻域平均（官方实现默认 True，本数据集实测 False 更好）
    normalize: bool = False                    # 特征是否逐维标准化（本数据集实测关掉更好）
    coreset_ratio: float = 0.01
    max_coreset: int = 6000
    n_neighbors: int = 1
    gaussian_sigma: float = 4.0
    batch_size: int = 8
    num_workers: int = 0
    device: str = "auto"


@dataclass
class ReportConfig:
    line_name: str = "金属螺母（metal_nut）产线 / 视觉质检工位 A2"
    # 判级逻辑（只做「相对自身位形」的判定，避免工件间光照/位形差异被当成缺陷）：
    #   NG  ⇔ 异常分数 ≥ 图像阈值  AND  热力图核心显著性 ≥ 阈值（相对最大值的比例）
    #   Major ⇔ 缺陷面积 ≥ max_area_ratio；Minor ⇔ ≥ min_area_ratio；否则 Trace
    image_score_threshold: float = 0.5         # 兜底阈值（会被训练集标定值覆盖）
    defect_ratio_threshold: float = 0.002      # 异常面积占比下限（过小视为噪声）
    local_significance: float = 0.60           # 核心显著性门槛（区域峰值相对热力图峰值）
    max_area_ratio: float = 0.10               # 面积 ≥ 10% 判 Major
    min_area_ratio: float = 0.02               # 面积 ≥ 2% 判 Minor
    confidence_heavy: float = 0.7
    # 缺陷区域提取：在归一化热力图上取该百分位作为核心阈值（实测 88 兼顾召回与框紧）
    region_core_percentile: float = 88.0


@dataclass
class AgentConfig:
    provider: str = os.environ.get("QC_AGENT_PROVIDER", "auto")   # deepseek | openai | qwen | ollama | auto | none
    model: str = os.environ.get("QC_AGENT_MODEL", "")            # 空 = 用 provider 默认模型
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tool_rounds: int = 6
    recursion_limit: int = 24
    enable_llm: bool = True


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # ---------------------------------------------------------- 序列化
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AppConfig":
        d = d or {}
        return cls(data=DataConfig(**d.get("data", {})),
                   model=ModelConfig(**d.get("model", {})),
                   report=ReportConfig(**d.get("report", {})),
                   agent=AgentConfig(**d.get("agent", {})))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def dump_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    # ---------------------------------------------------------- 路径
    @property
    def dataset_dir(self) -> Path:
        return Path(self.data.root) / self.data.dataset

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / self.data.train_split

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / self.data.test_dir

    @property
    def gt_root(self) -> Path:
        return self.dataset_dir / self.data.gt_dir

    @property
    def model_tag(self) -> str:
        m = self.model
        bits = [m.backbone, f"sz{m.image_size}"]
        if m.auto_crop:
            bits.append("crop")
        if m.local_pool:
            bits.append("3x3")
        return "_".join(bits)

    def model_path(self) -> Path:
        return MODEL_DIR / f"{self.data.dataset}_{self.model_tag}.pt"


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    if path is not None:
        return AppConfig.from_yaml(path)
    default_yaml = PROJECT_ROOT / "configs" / "default.yaml"
    if default_yaml.exists():
        return AppConfig.from_yaml(default_yaml)
    return AppConfig()
