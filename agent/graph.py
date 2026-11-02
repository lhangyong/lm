"""LangGraph 质检智能体。

图结构（state 为 QCState）::

    START ─> planner ─> agent ─┬─ tool_calls ─> tools ─┐
                               │                       │
                               └──────────────> reporter ─> render ─> END
                                     ↑ (tools 回到 agent 继续推理)

节点职责：
1. planner   规则解析请求（图片 / 目录 / 普通问答），生成确定性质检计划；
2. agent     工具调用节点：有 tool calling 时走原生 ReAct 循环，否则「LLM 出 JSON 计划 → 本地执行」，
             完全没有 LLM 时走纯规则执行；
3. reporter  只依据 state 中真实工具输出（evidence）写解读，无 LLM 时用确定性模板；
4. render    把解读 + 真实数据渲染成 Markdown / HTML 报告落盘。
"""
from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from patchcore_qc.config import REPORT_DIR, AppConfig
from patchcore_qc.detector import DefectInspector, strip_internal

from . import report as report_render
from .llm import LLMStatus, build_llm
from .tools_data import (DATA_TOOLS, dataset_overview, list_datasets, list_models, read_eval_report,
                         set_context as set_data_context)
from .tools_detect import (DETECTION_TOOLS, calibrate_thresholds, explain_defect_classes, inspect_folder,
                           inspect_image, set_context as set_detect_context, set_threshold, split_tool_output)

__all__ = ["QCAgent", "build_graph", "create_agent", "ALL_TOOLS", "extract_image_paths",
           "extract_folder_paths", "inspect_image", "inspect_folder", "calibrate_thresholds", "set_threshold",
           "explain_defect_classes", "list_datasets", "dataset_overview", "list_models", "read_eval_report"]

try:
    from langgraph.prebuilt import ToolNode
except Exception:  # pragma: no cover
    ToolNode = None  # type: ignore

ALL_TOOLS: List[BaseTool] = DETECTION_TOOLS + DATA_TOOLS
TOOL_MAP: Dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


# ================================================================ 提示词
SYSTEM_PROMPT = """你是一名工业视觉质检工程师，在操作 PatchCore 缺陷检测系统。

工作规则：
1. 必须先调用工具拿到真实检测数据，禁止编造分数、缺陷区域、文件名或指标。
2. 用户给出图片路径用 inspect_image；给出目录用 inspect_folder（必要时加 limit）；
   询问模型/数据集/评估情况用 list_models、dataset_overview、read_eval_report。
3. 路径必须是本机绝对路径；不熟悉的路径先 list_datasets / dataset_overview 了解结构。
4. 拿到数据后，用中文给出：判定 OK/NG、严重度、缺陷位置、处置建议（放行/隔离/拦截）。
5. 不要编造百分比与因果解释。"""

PLAN_PROMPT = """你是质检调度器。根据用户请求规划工具调用，只输出一行 JSON：
{{"tool_calls":[{{"name":"工具名","args":{{}}}}],"rationale":"理由"}}

可用工具：
{tools}

上下文：数据集目录 {dataset_dir}；训练集 {train_dir}；测试集 {test_dir}
用户请求：{request}"""

SUMMARY_PROMPT = """你是工业质检报告撰写员。根据下列「检测证据」写质检解读（中文 Markdown，200~350 字）：

1. 结论：OK/NG、严重度、判定依据（异常分数 vs 阈值）；
2. 缺陷定位：只引用证据中出现的位置、面积占比、峰值；
3. 可能成因：用「可能」措辞，结合该类缺陷的常见工艺原因；
4. 处置建议：放行 / 隔离复检 / 拦截返修；
5. 最后一行写「数据来源：工具名」。

检测证据：
{evidence}

用户请求：{request}"""


# ================================================================ 状态
class QCState(TypedDict, total=False):
    request: str
    profile: str                                                    # single | multi | folder | general
    targets: List[str]
    plan: str
    messages: Annotated[List[AnyMessage], operator.add]
    inspector_messages: Annotated[List[AnyMessage], operator.add]
    evidence: Annotated[List[Dict[str, Any]], operator.add]
    rounds: int
    analysis: str
    report_markdown: str
    report_html: str
    report_path: str
    llm_status: Dict[str, Any]
    trace: Annotated[List[Dict[str, Any]], operator.add]


@dataclass
class AgentContext:
    cfg: AppConfig
    inspector: DefectInspector
    llm: Any = None
    status: LLMStatus = field(default_factory=LLMStatus)
    max_tool_rounds: int = 6
    report_dir: Path = REPORT_DIR

    @property
    def use_llm(self) -> bool:
        return self.llm is not None


# ================================================================ 路径解析
_IMAGE_RE = re.compile(r'([A-Za-z]:\\[^\s"\'<>,;|*?]*?\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))', re.IGNORECASE)
_DIR_RE = re.compile(r'([A-Za-z]:\\[^\s"\'<>,;|*?]+)')
_QUOTED_RE = re.compile(r'["“]([A-Za-z]:\\[^"”]+)["”]')
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def texts_of(msg: Any) -> str:
    """把消息内容统一成纯文本。"""
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    try:
        return json.dumps(c, ensure_ascii=False)
    except Exception:
        return str(c)


def extract_image_paths(text: str) -> List[str]:
    cands = _IMAGE_RE.findall(text) + _QUOTED_RE.findall(text)
    out = []
    for c in cands:
        p = Path(c.strip())
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.append(str(p))
    return list(dict.fromkeys(out))


def extract_folder_paths(text: str) -> List[str]:
    out = []
    for c in _DIR_RE.findall(text) + _QUOTED_RE.findall(text):
        p = Path(c.strip().rstrip("\\"))
        if p.is_dir():
            out.append(str(p))
    return list(dict.fromkeys(out))


# ================================================================ 节点
def make_planner(ctx: AgentContext):
    def planner(state: QCState) -> Dict[str, Any]:
        request = state.get("request", "")
        images = extract_image_paths(request)
        img_dirs = {str(Path(i).parent) for i in images}
        folders = [f for f in extract_folder_paths(request) if f not in img_dirs]
        if images:
            profile = "single" if len(images) == 1 else "multi"
            targets = images
        elif folders:
            profile, targets = "folder", folders[:3]
        else:
            profile, targets = "general", []
        plan = {"single": f"inspect_image({targets[0] if targets else '-'}) → 判定 → 出具报告",
                "multi": f"逐张 inspect_image（{len(targets)} 张）→ 汇总 → 报告",
                "folder": f"inspect_folder({targets[0] if targets else '-'}) → 统计 → 汇总报告",
                "general": "查询数据集/模型/评估信息 → 回答"}[profile]
        return {"profile": profile, "targets": targets, "plan": plan, "rounds": 0,
                "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request)],
                "trace": [{"node": "planner", "profile": profile, "plan": plan, "targets": targets}]}
    return planner


def make_agent(ctx: AgentContext):
    def agent(state: QCState) -> Dict[str, Any]:
        rounds = int(state.get("rounds", 0)) + 1
        msgs = state.get("inspector_messages") or []
        use_native = ctx.use_llm and ctx.status.tools

        if rounds == 1 and not use_native and state.get("profile") == "single" and len(state.get("targets", [])) == 1:
            out = _run_tools([("inspect_image", {"image_path": state["targets"][0], "save_visualization": True})])
            return {"inspector_messages": out["messages"], "evidence": out["evidence"], "rounds": rounds,
                    "trace": [{"node": "agent", "mode": "rule_direct", "tools": out["tools"]}]}

        if use_native:
            msgs = msgs or state.get("messages") or [HumanMessage(content=state.get("request", ""))]
            try:
                ai = ctx.llm.invoke(msgs)
                calls = [c.get("name") for c in (getattr(ai, "tool_calls", None) or [])]
                return {"inspector_messages": [ai], "rounds": rounds,
                        "trace": [{"node": "agent", "mode": "llm_tool_calling", "tool_calls": calls}]}
            except Exception as e:
                fb = _fallback_execution(ctx, state, use_llm=False)
                fb["rounds"] = rounds
                fb["trace"][0]["error"] = str(e)[:200]
                return fb

        if rounds > ctx.max_tool_rounds:
            return {"rounds": rounds, "trace": [{"node": "agent", "mode": "max_rounds"}]}
        fb = _fallback_execution(ctx, state, use_llm=ctx.use_llm)
        fb["rounds"] = rounds
        return fb
    return agent


def _run_tools(calls: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    evidence, messages = [], []
    for name, args in calls:
        try:
            text = TOOL_MAP[name].invoke(args)
        except Exception as e:
            text = f"工具 {name} 执行失败：{e}"
        summary, payload = split_tool_output(text)
        evidence.append({"tool": name, "args": args, "summary": summary, "payload": payload})
        messages.append(HumanMessage(content=f"[已执行 {name}] {summary}"))
    return {"evidence": evidence, "messages": messages, "tools": [c[0] for c in calls]}


def _fallback_execution(ctx: AgentContext, state: QCState, use_llm: bool) -> Dict[str, Any]:
    profile = state.get("profile", "general")
    targets = state.get("targets", [])
    calls: List[Tuple[str, Dict[str, Any]]] = []
    mode = "rule"
    if use_llm and ctx.use_llm:
        prompt = PLAN_PROMPT.format(
            tools="\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in ALL_TOOLS),
            dataset_dir=ctx.cfg.dataset_dir, train_dir=ctx.cfg.train_dir, test_dir=ctx.cfg.test_dir,
            request=state.get("request", ""))
        try:
            resp = ctx.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(_extract_json(texts_of(resp)))
            for c in (data.get("tool_calls") or [])[:6]:
                if c.get("name") in TOOL_MAP:
                    calls.append((c["name"], dict(c.get("args") or {})))
            mode = "llm_json_plan"
        except Exception:
            calls = []
    if not calls:
        mode = "rule"
        if profile in ("single", "multi"):
            calls = [("inspect_image", {"image_path": p, "save_visualization": True}) for p in targets[:6]]
        elif profile == "folder":
            m = re.search(r"(\d+)\s*张", state.get("request", ""))
            calls = [("inspect_folder", {"folder": targets[0], "limit": int(m.group(1)) if m else 12})]
        else:
            calls = [("dataset_overview", {}), ("list_models", {}), ("read_eval_report", {})]
    out = _run_tools(calls)
    return {"inspector_messages": out["messages"], "evidence": out["evidence"],
            "trace": [{"node": "agent", "mode": mode, "tools": out["tools"]}]}


def route_after_agent(ctx: AgentContext):
    def router(state: QCState) -> str:
        msgs = state.get("inspector_messages") or []
        last = msgs[-1] if msgs else None
        rounds = int(state.get("rounds", 0))
        if ctx.status.tools and isinstance(last, AIMessage) and getattr(last, "tool_calls", None) \
                and rounds < ctx.max_tool_rounds:
            return "tools"
        return "reporter"
    return router


def make_reporter(ctx: AgentContext):
    def reporter(state: QCState) -> Dict[str, Any]:
        evidence = state.get("evidence") or []
        items = collect_items(evidence)
        if ctx.use_llm and evidence:
            try:
                prompt = SUMMARY_PROMPT.format(evidence=digest(evidence), request=state.get("request", ""))
                resp = ctx.llm.invoke([SystemMessage(content="你是严谨的工业质检工程师，只依据给定数据作答。"),
                                       HumanMessage(content=prompt)])
                analysis = texts_of(resp)
            except Exception as e:
                analysis = f"（LLM 解读失败：{str(e)[:100]}）\n\n" + template_analysis(items, evidence)
        else:
            analysis = template_analysis(items, evidence)
        out = write_reports(ctx, items, analysis)
        return {"analysis": analysis.strip(), **out,
                "trace": [{"node": "reporter", "llm": ctx.use_llm, "report": out.get("report_path")}]}
    return reporter


def make_render(ctx: AgentContext):
    def render(state: QCState) -> Dict[str, Any]:
        return {"trace": [{"node": "render", "report": state.get("report_path")}]}
    return render


# ================================================================ 证据处理
def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text


def collect_items(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ev in evidence:
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("items"), list):
            items += [{**it, "_tool": ev["tool"]} for it in payload["items"] if "image_name" in it]
        elif "image_name" in payload:
            items.append({**payload, "_tool": ev["tool"]})
    return items


def digest(evidence: List[Dict[str, Any]], max_chars: int = 7000) -> str:
    blocks = []
    for ev in evidence:
        payload = ev.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            payload = {k: v for k, v in payload.items() if k != "items"}
            payload["ng_items"] = [i for i in ev["payload"]["items"] if i.get("verdict") == "NG"][:6]
        blocks.append(f"### 工具 {ev['tool']} 参数 {json.dumps(ev['args'], ensure_ascii=False)}\n"
                      f"摘要：{ev['summary']}\n数据：{json.dumps(payload, ensure_ascii=False)[:3500]}")
    return "\n\n".join(blocks)[:max_chars]


def _n(v: Any, d: int = 4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def template_analysis(items: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> str:
    """无 LLM 时的确定性解读（报告里所有数字都来自工具输出）。"""
    if not items:
        lines = ["### 结论", "本次请求没有产生单件图片的检测证据，不输出质检判定结论。", ""]
        lines += [f"**{ev['tool']}**：\n{ev['summary']}\n" for ev in evidence]
        lines += ["", "数据来源：" + "、".join(sorted({e["tool"] for e in evidence}))]
        return "\n".join(lines)
    ng = [i for i in items if i.get("verdict") == "NG"]
    lines = ["### 结论", f"共检测 {len(items)} 件：NG {len(ng)} 件，OK {len(items) - len(ng)} 件，"
                         f"不良率 {len(ng) / len(items):.1%}。", ""]
    for i in ng[:6]:
        why = []
        if i.get("ratio_vs_threshold", 0) >= 1:
            why.append(f"异常分数 {_n(i.get('anomaly_score'))} ≥ 阈值 {_n(i.get('threshold'))}"
                       f"（来源 {i.get('threshold_source', '-')}）")
        if i.get("defect_ratio", 0) >= i.get("defect_ratio_threshold", 1):
            why.append(f"异常面积占比 {i.get('defect_ratio', 0):.3%} ≥ 面积阈值 "
                       f"{i.get('defect_ratio_threshold', 0):.3%}")
        regs = i.get("regions") or []
        loc = "；".join(f"#{k + 1} {r.get('location_detail', r['location'])}（面积 {r['area_ratio']:.3%}，"
                        f"显著度 {r.get('peak_ratio', 1):.2f}×，{r['severity']}）"
                        for k, r in enumerate(regs[:3])) or "无非零连通区域"
        lines += [f"### {i['image_name']} · NG · {i.get('severity', '-')}",
                  f"- 判定依据：{'且'.join(why) or '综合判定'}",
                  f"- 缺陷定位：{loc}",
                  f"- 处置建议：{i.get('recommendation', '')}", ""]
    for i in [x for x in items if x.get("verdict") == "OK"][:3]:
        lines.append(f"- {i['image_name']}：OK，分数 {_n(i.get('anomaly_score'))} < 阈值 "
                     f"{_n(i.get('threshold'))}，可放行。")
    if ng:
        lines += ["", "可能成因（需结合工艺复核）：弯曲/翻转多与上料姿态、夹具压力有关；"
                      "变色多与镀层、氧化或光照漂移有关；划痕多与搬运、料道摩擦有关。"]
    lines += ["", "数据来源：" + "、".join(sorted({i.get("_tool", "inspect") for i in items}))]
    return "\n".join(lines)


def _bundle_visuals(item: Dict[str, Any], base: Path) -> Dict[str, Any]:
    """把可视化文件复制（或确认）到 base/assets 下，返回更新后的结果。"""
    visuals = item.get("visuals") or {}
    raw = Path(item.get("image_path", ""))
    if raw.exists():
        assets = base / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        dst = assets / f"{base.name}_{raw.stem}_source{raw.suffix.lower()}"
        try:
            import shutil
            if not dst.exists():
                shutil.copyfile(raw, dst)
            item = {**item, "image_path": str(dst)}
        except OSError:
            pass
    if not visuals:
        return item
    assets = base / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    new_visuals = {}
    for key, path in visuals.items():
        if not path:
            continue
        src = Path(path)
        dst = assets / src.name
        try:
            if src.resolve() != dst.resolve():
                import shutil
                shutil.copyfile(src, dst)
            new_visuals[key] = str(dst)
        except OSError:
            new_visuals[key] = str(src)
    return {**item, "visuals": new_visuals}


def write_reports(ctx: AgentContext, items: List[Dict[str, Any]], analysis: str) -> Dict[str, str]:
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"line_name": ctx.cfg.report.line_name}
    base = ctx.report_dir
    if len(items) == 1:
        item = strip_internal(items[0])
        # 可视化文件若在别的目录，先拷进报告 assets 目录，保证报告目录可整体拷走
        item = _bundle_visuals(item, base)
        md = report_render.render_markdown(item, meta=meta, analysis=analysis, base=base)
        html_text = report_render.render_html(item, meta=meta, analysis=analysis, base=base)
        stem = f"qc_{Path(item['image_name']).stem}_{stamp}"
    elif items:
        rows = []
        for i in items:
            row = strip_internal(i)
            rows.append(_bundle_visuals(row, base) if row.get("visuals") else row)
        md = report_render.render_batch_markdown(rows, meta=meta, analysis=analysis)
        html_text = report_render.render_batch_html(rows, meta=meta, analysis=analysis, base=base)
        stem = f"qc_batch_{stamp}"
    else:
        md = f"# 质检助手回答\n\n{analysis}\n"
        html_text = report_render.render_general_html(analysis, meta)
        stem = f"qc_summary_{stamp}"
    md_path, html_path = base / f"{stem}.md", base / f"{stem}.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return {"report_markdown": md, "report_path": str(md_path), "report_html": str(html_path)}


# ================================================================ 编译与入口
def build_graph(ctx: AgentContext):
    set_detect_context(ctx.inspector)
    set_data_context(ctx.cfg)
    g = StateGraph(QCState)
    g.add_node("planner", make_planner(ctx))
    g.add_node("agent", make_agent(ctx))
    g.add_node("reporter", make_reporter(ctx))
    g.add_node("render", make_render(ctx))
    g.add_edge(START, "planner")
    g.add_edge("planner", "agent")
    if ctx.status.tools and ToolNode is not None:
        g.add_node("tools", ToolNode(ALL_TOOLS, messages_key="inspector_messages"))
        g.add_edge("tools", "agent")
        g.add_conditional_edges("agent", route_after_agent(ctx), {"tools": "tools", "reporter": "reporter"})
    else:
        g.add_edge("agent", "reporter")
    g.add_edge("reporter", "render")
    g.add_edge("render", END)
    return g.compile()


@dataclass
class QCAgent:
    cfg: AppConfig
    inspector: DefectInspector
    enable_llm: bool = True
    verbose: bool = True

    def __post_init__(self):
        self.cfg.agent.enable_llm = self.enable_llm and self.cfg.agent.enable_llm
        bundle = build_llm(self.cfg.agent, ALL_TOOLS)
        self.status: LLMStatus = bundle.status
        self.ctx = AgentContext(cfg=self.cfg, inspector=self.inspector, llm=bundle.llm,
                                status=bundle.status, max_tool_rounds=self.cfg.agent.max_tool_rounds)
        self.graph = build_graph(self.ctx)
        if self.verbose:
            if self.status.enabled:
                print(f"[Agent] LLM: {self.status.provider}/{self.status.model} | "
                      f"tool calling={'on' if self.status.tools else 'off（JSON 计划模式）'}")
            else:
                print(f"[Agent] {self.status.reason}")

    def run(self, request: str) -> Dict[str, Any]:
        init: QCState = {"request": request, "messages": [], "inspector_messages": [], "evidence": [],
                         "trace": [], "llm_status": dict(self.status.__dict__)}
        final = self.graph.invoke(init, config={"recursion_limit": self.cfg.agent.recursion_limit})
        if self.verbose:
            for step in final.get("trace", []):
                print(f"  · {step}")
        return final


def create_agent(cfg: AppConfig, enable_llm: bool = True, verbose: bool = True) -> QCAgent:
    return QCAgent(cfg=cfg, inspector=DefectInspector(cfg), enable_llm=enable_llm, verbose=verbose)
