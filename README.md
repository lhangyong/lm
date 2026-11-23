# 金属螺母表面缺陷检测：PatchCore 无监督检测 + LangGraph 质检智能体

只用正常样本建模的工业质检方案：PatchCore 检出表面缺陷 → 分级与判型 → LangGraph 智能体出报告 → NG 进剔除队列，仿真环境下由机械臂分拣。

## 效果

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 测试集图像级 AUROC | 0.948 | 115 张测试图（good 22、bent 25、color 22、flip 23、scratch 23） |
| 测试集召回 | 91.4% | 漏检集中在弯曲、变色两类 |
| 训练成本 | 220 张正常图，单卡约 21 s | 无需缺陷样本，无需标注 |
| 换型成本 | 秒级 | 只重建记忆库，不重标数据 |

> 指标随数据批次会波动，仓库不含数据集，复现请按下方步骤自备 MVTec AD 的 metal_nut 子集。

## 方案

1. **检测**：PatchCore（CVPR 2022）只用正常样本构建记忆库，ResNet18 骨干 + 邻域聚合，输出异常分数与像素级热力图。
2. **定位与分级**：热力图经连通域给出缺陷位置、面积与严重度（Major / Minor / Trace），阈值按品类维护。
3. **判型与报告**：LangGraph 状态图 + 大模型（OpenAI 兼容接口）+ 9 个工具，把检测结果、工艺规则与批次统计串成 Markdown / HTML / Word 质检报告；也可用 `--no-llm` 走纯规则引擎离线出报告。
4. **执行**：NG 进剔除队列，机械臂状态机完成取料—吸盘—横移—放料，预留 Modbus TCP 点表，防护门与急停作为状态机约束（当前为仿真对象，真机未联调）。

## 目录结构

```
patchcore_qc/    检测核心：PatchCore 训练、评估、指标与可视化
agent/           LangGraph 质检智能体、报告生成与 CLI
robot_sorting/   机械臂分拣仿真与状态机
web/             网页版（检测 → 报告 → 分拣一屏操作）
configs/         模型、阈值与运行配置
scripts/         训练、评估、自检脚本
examples/        最小调用示例（检测 / 智能体）
docs/            设计与说明文档（端侧部署与量化、机械臂分拣等）
tools/           辅助工具脚本
```

## 快速开始

环境要求：Python 3.10、PyTorch（CUDA 可选）、其余依赖见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

### 1. 准备数据

从 MVTec AD 官网下载 `metal_nut` 子集，解压后保持如下结构（数据集不入库）：

```
data/metal_nut/
├── train/good/          # 220 张正常图，仅训练用
├── test/                # good / bent / color / flip / scratch
└── ground_truth/        # 像素级标注，仅评估用
```

### 2. 训练

```bash
python scripts/train.py
```

产出记忆库、标定阈值与运行配置到 `outputs/`。

### 3. 评估

```bash
python scripts/evaluate.py
```

输出图像级 / 像素级 AUROC、PRO / AUPRO、阈值混淆矩阵、逐类别 AUROC 与阈值灵敏度，可视化图落在 `outputs/visualizations/`。

### 4. 网页版

```bash
python -m web.server        # 打开 http://127.0.0.1:7860
```

### 5. 自检

```bash
python scripts/smoke_test.py
```

一次跑通「模型 → 检测器 → 智能体 → 报告」全链路。

## 质检智能体

```bash
# 单图质检，出 Markdown + HTML 报告
python -m agent.cli "检测 data/metal_nut/test/scratch/000.png 并出具质检报告"

# 整批复检
python -m agent.cli "复检 data/metal_nut/test/flip 这个目录，最多 8 张"

# 查询模型与评估情况
python -m agent.cli "当前模型的 AUROC、阈值和混淆矩阵是什么？"

# 不调用大模型，走离线规则引擎
python -m agent.cli "..." --no-llm

# 交互模式
python -m agent.cli --interactive
```

大模型接入走环境变量（OpenAI 兼容接口，默认 DeepSeek），不配置时自动降级为规则引擎。

## 说明与限制

- 仓库不含数据集与训练产物，`outputs/`、`data/` 需自行生成。
- 分拣执行端当前为仿真对象，真机联调尚未启动；Modbus TCP 点表已预留。
- 阈值随品类单独标定，换型只重建记忆库；跨批次精度会随光照与纹理变化波动，正式产线需按批次复核。
- 本项目为工业视觉方向的学习与验证工程，检测指标以仓库内评估脚本输出为准。
