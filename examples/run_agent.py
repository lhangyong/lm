# 质检智能体（LangGraph + 大模型 + PatchCore 工具）

```python
from patchcore_qc.config import load_config
from agent import create_agent

cfg = load_config()
agent = create_agent(cfg)                       # 自动读取 DEEPSEEK_API_KEY 等环境变量
final = agent.run(r"请检测 E:\Projects\shixun\jslm\metal_nut\test\bent\000.png 并出具质检报告")

print(final["analysis"])        # 大模型（或规则模板）写的质检解读
print(final["report_path"])     # outputs/reports/qc_000_xxx.md
print(final["report_html"])     # outputs/reports/qc_000_xxx.html
print(final["trace"])           # 每个 LangGraph 节点的执行轨迹
print(final["evidence"])        # 工具原始输出（报告中的数字全部来自这里）
```
