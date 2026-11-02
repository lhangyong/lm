"""端到端自测：训练产物是否齐全、检测器/评估/智能体三条链路是否可用。

用法： python scripts/smoke_test.py [--image 某张图] [--no-llm]
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 依赖统一装在项目内 .deps（conda 环境只读），自动引入，保证脚本可单独运行
_deps = Path(__file__).resolve().parents[1] / ".deps"
if _deps.is_dir():
    sys.path.insert(0, str(_deps))

from patchcore_qc.config import load_config
from patchcore_qc.dataset import build_test_items, list_images
from patchcore_qc.detector import DefectInspector


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--no-llm", action="store_true")
    args = p.parse_args()

    ok = True
    cfg = load_config(args.config)
    print(f"[1/4] 配置: 数据集={cfg.data.dataset} 模型={cfg.model_path().name}")
    if not cfg.model_path().exists():
        print("      ✗ 模型不存在，请先运行 scripts/train.py")
        return 1

    print("[2/4] 检测器")
    try:
        ins = DefectInspector(cfg)
        print(f"      ✓ 模型加载成功，阈值 {ins.thresholds.image_score:.4f}（{ins.thresholds.source}）")
    except Exception:
        traceback.print_exc()
        return 1

    print("[3/4] 单图推理")
    img = args.image
    if img is None:
        files = list_images(cfg.test_dir / "scratch")
        img = str(files[0]) if files else str(build_test_items(cfg)[0].path)
    try:
        r = ins.inspect_image(img, out_dir=None)
        print(f"      ✓ {r['image_name']}: {r['verdict']} 分数={r['anomaly_score']:.4f} "
              f"严重度={r['severity']} 区域={r['num_regions']} 用时={r['inference_ms']}ms")
    except Exception:
        traceback.print_exc()
        ok = False

    print("[4/4] 智能体（LangGraph）")
    try:
        from agent.graph import create_agent
        ag = create_agent(cfg, enable_llm=not args.no_llm, verbose=False)
        final = ag.run(f"请质检这张图片并出具报告：{img}")
        print(f"      ✓ 路径: {[t.get('node') for t in final.get('trace', [])]}")
        print(f"      ✓ 报告: {final.get('report_path')}")
        print(f"      ✓ HTML: {final.get('report_html')}")
    except Exception:
        traceback.print_exc()
        ok = False

    print("\n全部通过 ✅" if ok else "\n存在失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
