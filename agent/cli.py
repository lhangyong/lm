"""质检智能体命令行入口。

用法::

    # 单图质检（生成 Markdown + HTML 报告）
    python -m agent.cli "请检测 E:\\Projects\\shixun\\jslm\\metal_nut\\test\\scratch\\000.png 并出报告"

    # 整批复检（目录）
    python -m agent.cli "复检 E:\\Projects\\shixun\\jslm\\metal_nut\\test\\flip 这个目录，最多 8 张"

    # 问模型/评估情况
    python -m agent.cli "当前模型的 AUROC 和阈值是多少？"

    # 关闭大模型，纯规则引擎
    python -m agent.cli "..." --no-llm

也可以进入交互模式： python -m agent.cli --interactive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import create_agent
from patchcore_qc.config import load_config
from patchcore_qc.detector import DefectInspector, resolve_thresholds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PatchCore 质检智能体（LangGraph + 大模型）")
    p.add_argument("request", nargs="?", default="", help="自然语言请求（含图片/目录绝对路径）")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--provider", default=None, choices=["auto", "deepseek", "openai", "qwen", "ollama", "none"])
    p.add_argument("--model", default=None, help="例如 deepseek-chat / gpt-4o-mini / qwen-plus")
    p.add_argument("--base-url", default=None)
    p.add_argument("--no-llm", action="store_true", help="不调用大模型，使用离线规则引擎")
    p.add_argument("--threshold", type=float, default=None, help="手动指定图像级阈值")
    p.add_argument("--interactive", action="store_true", help="进入多轮交互模式")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.dataset:
        cfg.data.dataset = args.dataset
    if args.provider:
        cfg.agent.provider = args.provider
    if args.model:
        cfg.agent.model = args.model
    if args.base_url:
        cfg.agent.base_url = args.base_url
    if args.no_llm:
        cfg.agent.enable_llm = False

    inspector = DefectInspector(cfg)
    if args.threshold is not None:
        inspector.thresholds = resolve_thresholds(cfg, inspector.model, {"image_score_threshold": args.threshold})
    agent = create_agent(cfg, enable_llm=cfg.agent.enable_llm, verbose=not args.quiet)

    def handle(request: str) -> None:
        final = agent.run(request)
        print("\n========== 智能体解读 ==========")
        print(final.get("analysis", ""))
        print("\n========== 报告文件 ==========")
        print("Markdown:", final.get("report_path"))
        print("HTML    :", final.get("report_html"))

    if args.interactive or not args.request:
        print("输入质检请求（含图片/目录绝对路径），输入 q 退出。")
        while True:
            try:
                request = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if request.lower() in ("q", "quit", "exit", ""):
                break
            handle(request)
        return 0
    handle(args.request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
