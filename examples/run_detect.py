# 直接用 PatchCore 做单图 / 批量检测（不经过智能体）

```python
from patchcore_qc.config import load_config
from patchcore_qc.detector import DefectInspector

cfg = load_config()                    # 读取 configs/default.yaml
inspector = DefectInspector(cfg)       # 加载 outputs/models/*.pt + 标定阈值

result = inspector.inspect_image(r"E:\Projects\shixun\jslm\metal_nut\test\flip\000.png",
                                 out_dir=r"E:\Projects\shixun\jslm\outputs\visualizations")
print(result["verdict"], result["anomaly_score"], result["threshold"], result["severity"])
print(result["regions"])               # 缺陷区域：位置(九宫格)、面积占比、热力图峰值
print(result["visuals"])               # 叠加图 PNG / 热力图 SVG / 分数直方图

# 批量
batch = inspector.inspect_folder(r"E:\Projects\shixun\jslm\metal_nut\test\scratch", limit=10)
print(sum(r["verdict"] == "NG" for r in batch), "/", len(batch), "NG")
```
