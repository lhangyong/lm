# PatchCore 无监督缺陷检测 + LangGraph 智能体质检 + 机械臂自动分拣

> 数据集：`E:\Projects\shixun\jslm\metal_nut`（MVTec-AD 金属螺母子集）
> 算法：PatchCore（Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022）——**只用正常样本训练**
> 智能体：LangGraph 状态图 + 大模型（OpenAI 兼容接口，默认 DeepSeek）+ 9 个工具，输出 Markdown / HTML / Word 质检报告
> 执行：检测出 NG 后自动进入剔除队列，机械臂抓取放入废料箱；仿真对象 + 可替换 Modbus 驱动

---

## 0. 项目位置

工程和数据并排放在同一个目录下：

```
E:\Projects\shixun\jslm\
├── metal_nut\          # 数据集（原样保留，只读使用）
│   ├── train\good\     # 220 张正常图（训练用）
│   ├── test\           # good 22 / bent 25 / color 22 / flip 23 / scratch 23，共 115 张
│   └── ground_truth\   # 缺陷像素标注（仅评估用）
├── agent\ patchcore_qc\ scripts\ configs\ web\ examples\   # 本项目全部代码
├── outputs\           # 模型、评估、可视化、报告
├── .deps\             # 运行依赖（本项目专用，不污染 conda 环境）
├── start_web.bat / train.bat / eval.bat / smoke_test.bat   # 一键脚本
└── README.md / requirements.txt
```

> 数据集体积大，**不纳入 git 仓库**：从 MVTec AD 官网/官方仓库下载 metal_nut 子集，解压放入 `metal_nut\` 目录即可。

---

## 1. 快速开始

### 一键脚本（和晶体管项目同款，双击即用）

| 操作 | 脚本 | 说明 |
|---|---|---|
| 1 训练 | 双击 `train.bat` | 用 `train\good` 正常样本训练 → 输出模型 + 标定阈值 |
| 2 评估 | 双击 `eval.bat` | 测试集 115 张 AUROC / 混淆矩阵 → `outputs\` |
| 3 网页 | 双击 `start_web.bat` | 启动网页版并自动打开浏览器 `http://127.0.0.1:7860` |
| 4 自检 | 双击 `smoke_test.bat` | 一键验证模型/检测器/智能体链路 |

> 依赖装在项目内 `.deps`（conda 环境目录只读），三个脚本已通过 `PYTHONPATH` 自动引入，直接双击即可。

### 1.1 环境（已在本机验证）

```
Python     : D:\anconda\envs\yolov8-gpu\python.exe  (Python 3.10)
torch      : 2.14.0+cu126      torchvision 0.29.0+cu126
numpy      : 2.2.6             opencv-python 5.0.0      scikit-learn 1.7.2
langgraph  : 1.2.16            langchain-core 1.6.12    langchain-openai 1.7.2
```

换新机器时安装依赖：

```powershell
D:\anconda\envs\yolov8-gpu\python.exe -m pip install -r requirements.txt
```

> 首次运行会自动下载 ImageNet 预训练权重 `resnet18-f37072fd.pth`（约 45MB，缓存在 `%USERPROFILE%\.cache\torch`）。

### 1.2 训练（无监督，只用 `train/good` 的 220 张正常图，约 21 秒）

```powershell
cd E:\Projects\shixun\jslm
D:\anconda\envs\yolov8-gpu\python.exe scripts/train.py
```

产出：`outputs/models/metal_nut_resnet18_sz256_3x3.pt`（记忆库）、`.calibration.json`（标定阈值）、`.config.yaml`。

### 1.3 评估（测试集 115 张）

```powershell
D:\anconda\envs\yolov8-gpu\python.exe scripts/evaluate.py
```

输出图像级 AUROC、像素级 AUROC、PRO / AUPRO@0.3、阈值混淆矩阵、逐类别 AUROC、阈值灵敏度，并把叠加图/热力图写到 `outputs/visualizations/eval/`。

### 1.4 运行质检智能体

```powershell
# ① 单图质检 → 生成 Markdown + HTML 报告
D:\anconda\envs\yolov8-gpu\python.exe -m agent.cli "请检测 E:\Projects\shixun\jslm\metal_nut\test\scratch\000.png 并出具质检报告"

# ② 整批复检（目录）
D:\anconda\envs\yolov8-gpu\python.exe -m agent.cli "复检 E:\Projects\shixun\jslm\metal_nut\test\flip 这个目录，最多 8 张"

# ③ 问模型 / 数据集 / 评估情况
D:\anconda\envs\yolov8-gpu\python.exe -m agent.cli "当前模型的 AUROC、阈值和混淆矩阵是什么？"

# ④ 不调用大模型（离线规则引擎，同样出报告）
D:\anconda\envs\yolov8-gpu\python.exe -m agent.cli "..." --no-llm

# ⑤ 交互模式 / 一键自检
D:\anconda\envs\yolov8-gpu\python.exe -m agent.cli --interactive
D:\anconda\envs\yolov8-gpu\python.exe scripts/smoke_test.py
```

接入大模型（任选其一，设置环境变量即可，无需改代码）：

```powershell
$env:DEEPSEEK_API_KEY  = "sk-..."     # provider=deepseek，默认 deepseek-chat
# $env:OPENAI_API_KEY  = "sk-..."     # provider=openai，默认 gpt-4o-mini
# $env:DASHSCOPE_API_KEY = "sk-..."   # provider=qwen（DashScope 兼容模式），默认 qwen-plus
# 本地 Ollama：--provider ollama --model qwen2.5:7b
#              或 $env:QC_AGENT_BASE_URL="http://127.0.0.1:11434/v1"
```

**没有 API Key 时不会报错**：智能体自动降级为「离线规则引擎」，照样完成检测、判定、写报告，报告里注明数据来源。

### 1.5 网页版（拖放图片即检测，推荐演示用）

```powershell
cd E:\Projects\shixun\jslm
D:\anconda\envs\yolov8-gpu\python.exe -m agent.web
```

启动后会自动打开浏览器（默认 `http://127.0.0.1:7860/`），使用方式：

1. 把一张螺母图片**拖进左侧虚线框**（或点击选择文件、或 Ctrl+V 粘贴）；
2. 也可以点下方**「测试集示例」缩略图**直接检测，或点「🎲 随机示例」随机抽一张；
3. 右侧立刻显示：判定 OK/NG、严重度、异常分数/阈值、推理耗时、原图、异常叠加图、
   热力图、分数直方图、缺陷区域表；下方给出智能体解读；
4. 检测为 NG 时自动进入剔除队列，左侧“机械臂分拣”卡片显示机械臂状态、队列、已剔除数和放行数；
   可切换自动/监控、急停、复位；
5. 点「导出」并选择格式：HTML 可直接在浏览器查看；Word（.docx）带原图、叠加图、指标表和缺陷区域表；Markdown 便于继续编辑。文件都存到 `outputs/reports/`。

可选参数：`--port 8000` 换端口、`--no-browser` 不自动开浏览器、`--no-llm` 禁用大模型解读。
这是只监听本机 `127.0.0.1` 的轻量服务（Python 标准库实现，无额外依赖），按 Ctrl+C 停止。

---

## 2. 原理

### 2.1 PatchCore 检测流程

```
train/good 正常图 ─► ResNet18(layer2+layer3, 3x3 邻域聚合) ─► 32x32 个 patch × 384 维特征
                                                                     │
                                                       拼成 memory bank（22.5 万个 patch）
                                                                     │ 贪心 k-center 采样 1%
                                                                     ▼
                                                              coreset（2253 个 patch）
                                                                     ▲
test 图 ─► 同样的 patch 特征 ─► 每个 patch 到 coreset 的最近邻距离 ─► 高斯平滑热力图
                                                                     │
                                    图像分数 = 热力图最大值 ─► 与训练集标定阈值比较 ─► OK / NG
```

| 环节 | 本项目实现 | 说明 |
|---|---|---|
| 特征 | ResNet18 `layer2`+`layer3` 拼接（384 维） | 论文用 WideResNet50（1536 维）；可 `--backbone wide_resnet50_2` |
| 局部聚合 | 3×3 邻域平均（`local_pool: true`） | 与论文一致；实测对划痕这类局部纹路缺陷提升明显 |
| 特征标准化 | 关闭（`normalize: false`） | 实测开启后 AUROC 从 0.95 掉到 0.46，故关闭 |
| coreset | 贪心最远点采样 1%，大规模时随机子空间投影加速 | 论文 §4.1 |
| 重加权 | coreset 每个 patch 与最近 3 个 core patch 的平均距离 | 论文公式 (2)(3) |
| 打分 | 最近邻距离 → 高斯平滑 σ=4 → 取最大值 | 论文公式 (4)(5)(6) |
| 阈值 | 训练集正常样本 **max(P99.5, mean+3σ)**；不接触任何缺陷标注 | 报告里同时给出阈值灵敏度 |

### 2.2 LangGraph 智能体

```
START ─► planner ─► agent ─┬─(tool_calls)─► tools ─┐
                           │                       │
                           └──────────────► reporter ─► render ─► END
```

| 节点 | 职责 | 无大模型时 |
|---|---|---|
| **planner** | 解析请求里的图片/目录绝对路径，判定 single / multi / folder / general | ✅ |
| **agent** | 有 function calling → 原生 ReAct 循环（`ToolNode`）；否则 LLM 出 JSON 计划、本地执行；无 LLM → 规则执行 | ✅ |
| **reporter** | 只使用状态里的真实工具输出（evidence）写解读，绝不自己编数字 | ✅ 确定性模板 |
| **render** | 渲染 Markdown / HTML / Word 报告（原图、叠加图、热力图 SVG、分数直方图） | ✅ |

工具清单：

| 工具 | 作用 |
|---|---|
| `inspect_image` | 单图检测：判定、分数、缺陷区域（位置/面积/显著度/分级）、可视化 |
| `inspect_folder` | 目录批量检测：NG 数、不良率、逐张结果 |
| `calibrate_thresholds` | 用训练正常样本重新标定阈值 |
| `set_threshold` | 人工指定图像阈值 |
| `explain_defect_classes` | bent/color/flip/scratch 的工业语义 |
| `list_datasets` / `dataset_overview` | 只读查看数据根目录与类别分布 |
| `list_models` / `read_eval_report` | 查看已训练模型、最近一次评估指标 |

**代码对应**：`agent/graph.py` 里的 `QCState` 定义状态，`make_planner / make_agent / make_reporter / make_render`
是四个节点，`build_graph()` 连边，`QCAgent.run(request)` 跑完整条流水线并落盘报告。

### 2.3 机械臂自动分拣

```
检测结果 NG ─► 剔除队列 ─► 传送带送料 ─► 取料位下降 ─► 真空吸取
           ─► 提升 ─► 横移到废料箱 ─► 放下 ─► 回原点
```

- 状态机：`IDLE / CONVEY / APPROACH / DESCEND / GRIP / LIFT / TRANSFER / PLACE / RELEASE / RETURN`，
  另有 `ABORT / FAULT / ESTOP`。
- 安全：工作空间钳位、急停、防护门、真空确认、队列上限、单件超时、故障锁存。
- 仿真对象：`robot_sorting/cell.py`（三轴坐标 + 真空 + 料箱计数）。
- 可替换驱动：`robot_sorting/driver.py` 里的 `ModbusArmDriver` 骨架，真实机械臂按点表改 `REGISTER_MAP`。
- 网页接口：`/api/robot/status`、`/api/robot/mode`、`/api/robot/estop`、`/api/robot/reset`、`/api/robot/fault`。
- 验收：`robot_sorting/tests/test_sorting_loop.py`，7 项全过（NG 剔除、OK 放行、急停复位、监控模式、
  真空丢失中止、多件排队及队列满、无效结果转人工）。

> 注意：当前没有真实机械臂，Modbus 驱动未做现场连通测试；网页里是仿真动作，不能当作产线实测。
> 现场接入步骤见 `docs/11_机械臂自动分拣说明.md`。

---

## 3. 实测结果

> 环境：RTX 4060 Laptop 8GB / Python 3.10 / torch 2.14。测试集 115 张（good 22 + 缺陷 93），
> 阈值来自训练集正常样本自动标定，**未使用任何测试标签**。
> 指标定义：图像级 AUROC（good vs 缺陷）、像素级 AUROC（热力图逐像素，30% 权重采样）、
> PRO/AUPRO（连通域区域重叠，官方定义）。

| 指标 | 结果 |
|---|---|
| 图像级 AUROC | **0.9482** |
| 像素级 AUROC | **0.9584** |
| PRO / AUPRO | **0.8207** |
| 阈值 | 1.4594（训练集 P99.5 与 mean+3σ 的较大者） |
| 准确率 / Precision / Recall / F1 | 0.8957 / **0.9551** / **0.9140** / **0.9341** |
| 混淆矩阵 | TN 18，FP 4，FN 8，TP 85 |
| 逐类别 AUROC | bent 0.900，color 0.915，flip 1.000，scratch 0.980 |
| 单图推理（RTX 4060） | ≈ 27 ms |
| 训练耗时 | ≈ 21 s（220 张正常图 + 标定） |

**结果解读**：

- 漏检的 8 张全部是 `bent`（5 张）和 `color`（3 张），分数 1.38~1.45，只比阈值低 0~5%：
  这些是「轻微弯曲 / 轻微变色」的难样本，训练集正常图本身也存在位形和色差波动；
- 误报的 4 张全部是 `good`，分数 1.48~1.59（明显高于阈值）：属于正常件的极端个体；
- `flip`（翻转）100% 可分，`scratch`（划痕）0.98，说明局部纹理缺陷是 PatchCore 的强项；
- 严重度分布（按缺陷面积）：Major 10 张、Minor 79 张、其余为 Trace/None。

> 说明：判级在「图像分数 ≥ 阈值」之外，还要求存在局部显著异常区域；本数据集上这两种规则的
> P/R/F1 完全一致（TP 85 / FP 4 / FN 8），即区域机制不会牺牲精度，但能让报告给出定位。

**阈值灵敏度**（把阈值往上调 x%）：

| 阈值偏移 | 阈值 | Precision | Recall | F1 | NG 数 |
|---|---|---|---|---|---|
| 基准 | 1.4594 | 0.9551 | 0.9140 | 0.9341 | 89 |
| +1% | 1.4740 | 0.9540 | 0.8925 | 0.9222 | 87 |
| +2% | 1.4886 | 0.9643 | 0.8710 | 0.9153 | 84 |
| +3% | 1.5032 | 0.9753 | 0.8495 | 0.9080 | 81 |
| +5% | 1.5324 | 0.9863 | 0.7742 | 0.8675 | 73 |

> 产线可按「宁可多杀不可漏放」或「控制误报」自行上下调 1~3%。

---

## 4. 消融实验

同一训练集 / 同一测试集，每行只改一个因素：

| 配置 | 图像 AUROC | 像素 AUROC | PRO | F1 | 训练耗时 |
|---|---|---|---|---|---|
| **resnet18 · 256 · 3×3 聚合（默认）** | **0.9482** | **0.9584** | **0.8207** | **0.9341** | 21 s |
| resnet18 · 256 · 关闭 3×3 聚合 | 0.9218 | 0.9380 | 0.8012 | 0.8876 | 18 s |
| resnet18 · 256 · 开启特征标准化 | 0.4594 | 0.7364 | 0.6548 | 0.0825 | 17 s |
| resnet18 · 512 · 前景裁剪 | 0.9125 | 0.9380 | 0.8587 | 0.8824 | 181 s |

结论：

1. **3×3 邻域聚合（论文默认）在本数据集上是必要的**：关掉后 AUROC 掉约 0.026、F1 掉约 0.047；
2. **特征标准化会造成严重退化**（AUROC 0.46）：本数据集各 patch 特征尺度差异大，标准化后正常纹理的微小距离被放大；
3. **提高分辨率反而下降**（512 + 前景裁剪 0.9125 < 256 的 0.9482）：patch 过细后同一点位的正常纹理波动被当成异常，训练耗时还多 9 倍；
4. 因此默认配置取「256 + 3×3 聚合 + 不标准化」。

---

## 5. 报告示例

运行 1.4 的 ① 后生成 `outputs/reports/qc_<文件名>_<时间戳>.html`（可直接双击打开）与同名 `.md`：

- 顶部横幅：OK/NG 结论 + 异常分数 vs 阈值
- 基本信息：产线、模型、阈值来源、判定逻辑
- 可视化：原始图像、异常叠加图、热力图（SVG）、patch 分数分布
- 缺陷区域表：位置（左/右半部 - 上/下半部）、面积占比、显著度、分级
- 智能体分析：结论 / 判定依据 / 缺陷定位 / 可能成因 / 处置建议 / 数据来源
- 所有图片放在 `outputs/reports/assets/`，报告目录可整体拷走

严重度分级（`configs/default.yaml → report`）：

| 级别 | 条件 | 建议 |
|---|---|---|
| Major | 缺陷区域面积 ≥ 10% | 拦截，人工复检，评估返修或报废 |
| Minor | 缺陷区域面积 ≥ 2% | 隔离，人工抽检确认 |
| Trace | 其余非零区域 | 标记统计，可按工艺规程放行 |

> 判级规则：**NG ⇔ 异常分数 ≥ 图像阈值，且存在「局部显著」的缺陷区域**。缺陷区域提取方法：
> 先把热力图归一化到 [0,1]（1%~99% 分位截断），在像素分辨率上取 88 分位为阈值，
> 再按连通域统计面积/位置/峰值，得到的是缺陷的真实轮廓（见报告里的白边区域）。

---

## 6. 目录结构

```
E:\Projects\shixun\jslm\
├── configs/default.yaml          # 全部超参、阈值、provider
├── requirements.txt
├── README.md
├── start_web.bat / train.bat / eval.bat   # 一键脚本（双击即用）
├── patchcore_qc/
│   ├── config.py                 # AppConfig（路径/模型/报告/智能体）
│   ├── dataset.py                # 读图、几何对齐、前景裁剪、测试集与 GT 加载
│   ├── feature_extractor.py      # ResNet 多层特征 + patch 描述子
│   ├── patchcore.py              # 记忆库、coreset、kNN 打分、高斯平滑、后处理
│   ├── detector.py               # 阈值解析 + 单图/批量检测 → 结构化 JSON
│   ├── metrics.py                # AUROC / 像素AUROC / PRO@0.3 / 混淆矩阵 / 阈值灵敏度
│   └── viz.py                    # 叠加图 PNG、热力图 SVG、分数直方图 SVG
├── robot_sorting/                # 机械臂自动分拣
│   ├── cell.py                   # 三轴机械臂/传送带/料箱仿真
│   ├── safety.py                 # 工作空间/急停/防护门约束
│   ├── driver.py                 # 仿真驱动 + Modbus TCP 骨架
│   ├── controller.py             # 剔除队列 + 分拣状态机
│   └── tests/test_sorting_loop.py
├── agent/
│   ├── graph.py                  # LangGraph 状态图 planner→agent→reporter→render
│   ├── llm.py                    # provider 选择、ChatOpenAI 兼容封装、降级策略
│   ├── tools_detect.py           # 检测类工具
│   ├── tools_data.py             # 数据/模型/评估查询工具
│   ├── report.py                 # Markdown / HTML / Word 报告渲染
│   ├── cli.py                    # 命令行入口
│   └── web.py                    # 本地网页版服务（127.0.0.1:7860，自动开浏览器）
├── scripts/
│   ├── train.py                  # 训练（构建记忆库 + 标定）
│   ├── evaluate.py               # 评估
│   ├── smoke_test.py             # 一键自检（配置→检测器→推理→智能体）
│   └── _eval_model.py            # 评估任意模型文件
├── examples/
│   ├── run_agent.py              # 代码调用智能体
│   └── run_detect.py             # 直接用检测器
├── web/index.html                # 网页版前端（一屏全景）
├── .deps/                        # 运行依赖（本项目专用，不污染 conda 环境）
└── outputs/
    ├── models/                   # 模型、标定阈值、配置快照
    ├── runs/                     # 评估指标 JSON
    ├── visualizations/           # 叠加图、热力图、分数直方图
    └── reports/                  # 质检报告 .md / .html + assets/
```

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `FileNotFoundError: 未找到模型 ...` | 先运行 `scripts/train.py` |
| `CUDA out of memory` | 降 `--image-size` / `--max-coreset`，或 `--device cpu` |
| 首次下载权重失败 | 设置 `TORCH_HOME`，或手动放 `resnet18-f37072fd.pth` 到 `%USERPROFILE%\.cache\torch\hub\checkpoints` |
| 换产线/光照后误报增多 | 让智能体调用 `calibrate_thresholds`，或 `set_threshold` 人工收紧 |
| 换同结构数据集 | `--dataset 名称`，数据根目录用 `--data-root` 或环境变量 `PATCHCORE_DATA_ROOT` |
| 想要论文同款骨干 | `--backbone wide_resnet50_2`（显存约 6GB，训练更慢） |
