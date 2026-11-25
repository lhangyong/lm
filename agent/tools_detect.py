"""检测工具层：把 PatchCore 推理封装成 LangChain 工具，供 LangGraph 智能体调用。

工具输出统一格式：人类可读摘要 + ``<<<JSON>>>`` + 机器可读 JSON。
摘要进入 LLM 上下文用于解读；JSON 由报告节点直接取证，保证报告里的数字都有出处。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from patchcore_qc.config import VIS_DIR
from patchcore_qc.dataset import list_images
from patchcore_qc.detector import DefectInspector, jsonable, resolve_thresholds

_CTX: Dict[str, Any] = {}
JSON_SEP = "\n<<<JSON>>>\n"


def set_context(inspector: DefectInspector) -> None:
    _CTX["inspector"] = inspector


def get_inspector() -> DefectInspector:
    if "inspector" not in _CTX:
        raise RuntimeError("检测器尚未初始化（agent.tools_detect.set_context）")
    return _CTX["inspector"]


def split_tool_output(text: str) -> tuple[str, Optional[dict]]:
    if JSON_SEP in text:
        head, tail = text.split(JSON_SEP, 1)
        try:
            return head.strip(), json.loads(tail)
        except json.JSONDecodeError:
            return head.strip(), None
    return text.strip(), None


def pack(summary: str, payload: Dict[str, Any]) -> str:
    return summary.strip() + JSON_SEP + json.dumps(payload, ensure_ascii=False, default=str)


def _vis_dir() -> Path:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    return VIS_DIR


@tool
def inspect_image(image_path: str, save_visualization: bool = True) -> str:
    """对单张本地图片做 PatchCore 缺陷检测，返回判级、异常分数、缺陷位置与可视化文件。

    Args:
        image_path: 图片路径，绝对路径或相对仓库根目录，例如 metal_nut/test/scratch/000.png
        save_visualization: 是否保存叠加图/热力图/分数直方图
    """
    out_dir = _vis_dir() if save_visualization else None
    result = get_inspector().inspect_image(image_path, out_dir=out_dir, save_visual=save_visualization)
    return pack(summary_one(result), jsonable(result))


@tool
def inspect_folder(folder: str, limit: int = 12, save_visualization: bool = False) -> str:
    """对目录下图片批量检测，返回 NG 数量、不良率与逐张结果（适合整批抽检）。

    Args:
        folder: 图片目录路径，绝对路径或相对仓库根目录，例如 metal_nut/test/scratch
        limit: 最多检测多少张（默认 12）
        save_visualization: 是否保存可视化文件（最多前 6 张）
    """
    files = list_images(Path(folder))
    if not files:
        return pack(f"目录中没有图片: {folder}", {"error": "empty_folder", "folder": folder})
    out_dir = _vis_dir() if save_visualization else None
    results = get_inspector().inspect_many(files[:max(1, limit)], out_dir=out_dir, max_visuals=6)
    ng = [r for r in results if r["verdict"] == "NG"]
    payload = {
        "folder": folder,
        "count": len(results),
        "ng_count": len(ng),
        "ok_count": len(results) - len(ng),
        "ng_rate": round(len(ng) / len(results), 4),
        "ng_images": sorted(r["image_name"] for r in results if r["verdict"] == "NG"),
        "severity_hist": severity_hist(results),
        "items": [jsonable(r) for r in results],
    }
    summary = f"目录 {Path(folder).name}：检测 {len(results)} 张，NG {len(ng)} 张（{len(ng) / len(results):.1%}）"
    if ng:
        summary += "；NG: " + "、".join(payload["ng_images"][:8])
    return pack(summary, payload)


@tool
def calibrate_thresholds() -> str:
    """用训练集正常样本重新标定判定阈值（图像阈值 = max(P99.5, mean+3σ)）。换产线或重训模型后建议调用。"""
    inspector = get_inspector()
    cal = inspector.calibrate()
    t = inspector.thresholds
    summary = (f"标定完成：图像阈值 {t.image_score:.4f}（{t.source}），像素阈值 {t.pixel_score:.4f}；"
               f"正常样本分数 mean={cal['train_mean']:.4f} std={cal['train_std']:.4f} max={cal['train_max']:.4f}")
    return pack(summary, {"thresholds": {"image_score": t.image_score, "pixel_score": t.pixel_score,
                                         "defect_ratio": t.defect_ratio, "source": t.source},
                          "calibration": cal})


@tool
def set_threshold(image_score_threshold: float) -> str:
    """人工指定图像级判定阈值（覆盖自动标定，用于收紧或放宽误报）。

    Args:
        image_score_threshold: 新的图像分数阈值（正数，通常略高于正常样本最大分数）
    """
    if image_score_threshold <= 0:
        return pack("阈值必须为正数", {"error": "invalid_threshold"})
    inspector = get_inspector()
    inspector.thresholds = resolve_thresholds(inspector.cfg, inspector.model,
                                              {"image_score_threshold": image_score_threshold})
    t = inspector.thresholds
    return pack(f"已设置图像阈值 {t.image_score:.4f}（来源 {t.source}）",
                {"thresholds": {"image_score": t.image_score, "pixel_score": t.pixel_score,
                                "source": t.source}})


@tool
def explain_defect_classes(classes: str = "") -> str:
    """给出缺陷类别的工业语义解释（弯曲/变色/翻转/划痕），供写报告时引用。

    Args:
        classes: 逗号分隔的类别名；为空返回全部
    """
    from patchcore_qc.detector import DEFECT_SEMANTICS
    wanted = [c.strip() for c in classes.split(",")] if classes else list(DEFECT_SEMANTICS)
    wanted = [c for c in wanted if c]
    lines = [f"- {c}: {DEFECT_SEMANTICS.get(c, '未知类别（按通用外观缺陷处理）')}" for c in wanted]
    return pack("缺陷类别语义:\n" + "\n".join(lines),
                {"semantics": {c: DEFECT_SEMANTICS.get(c) for c in wanted}})


def severity_hist(results: List[Dict]) -> Dict[str, int]:
    hist = {"Major": 0, "Minor": 0, "Trace": 0, "None": 0}
    for r in results:
        hist[r["severity"]] = hist.get(r["severity"], 0) + 1
    return hist


def summary_one(r: Dict) -> str:
    lines = [f"{r['image_name']}: {r['verdict']}  异常分数={r['anomaly_score']:.4f}"
             f"（阈值 {r['threshold']:.4f}，来源 {r['threshold_source']}）"]
    if r["verdict"] == "OK":
        lines.append("未检出超阈值缺陷，建议放行")
    else:
        lines.append(f"严重度 {r['severity']}，缺陷区域 {r['num_regions']} 处，异常面积占比 {r['defect_ratio']:.4%}")
        for i, reg in enumerate(r["regions"][:5]):
            lines.append(f"  #{i + 1} {reg.get('location_detail', reg['location'])}：面积 {reg['area_ratio']:.4%}，"
                         f"显著度 {reg.get('peak_ratio', 1.0):.2f}×，级别 {reg['severity']}")
        lines.append(f"建议: {r['recommendation']}")
    if r.get("visuals"):
        lines.append("可视化: " + ", ".join(Path(v).name for v in r["visuals"].values()))
    return "\n".join(lines)


DETECTION_TOOLS = [inspect_image, inspect_folder, calibrate_thresholds, set_threshold, explain_defect_classes]
