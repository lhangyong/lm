"""本地网页版质检工具：在浏览器里拖放一张图片，点击检测即可看到结果。

启动::

    cd E:\\Projects\\shixun\\jslm
    D:\\anconda\\envs\\yolov8-gpu\\python.exe -m agent.web
    # 然后浏览器打开 http://127.0.0.1:7860

功能：
- 拖放 / 点击选择 / 直接粘贴图片，或从下方测试集示例中点选（方便随机抽样演示）
- 点击「开始检测」：PatchCore 推理 → 判定 OK/NG → 显示叠加图、热力图、缺陷区域表
- 可切换「大模型解读」：接入时由 LangGraph 智能体生成解读，未接入时用规则模板
- 一键把本次结果导出成离线报告（HTML / Word / Markdown）

说明：这是本机 127.0.0.1 上的轻量服务，不对外网开放，也不需要额外安装包
（只用 Python 标准库 http.server 与现有依赖）。
"""
from __future__ import annotations

import base64
import cgi
import json
import random
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from agent import report as report_render
from agent.graph import _bundle_visuals
from patchcore_qc.config import REPORT_DIR, VIS_DIR, AppConfig, load_config
from patchcore_qc.dataset import IMG_EXTS, list_images, load_image
from patchcore_qc.detector import DefectInspector, strip_internal
from robot_sorting.controller import RobotController

WEB_DIR = PROJECT_ROOT / "web"
WEB_UPLOADS = VIS_DIR / "web_uploads"
IMG_PATTERN = re.compile(r"\.(png|jpe?g|bmp|tif|tiff|webp)$", re.IGNORECASE)


# ================================================================ 状态
class WebState:
    """进程内共享状态：检测器只加载一次（模型 ~9MB，加载后常驻显存/内存）。"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.inspector: Optional[DefectInspector] = None
        self.agent = None
        self.load_error: Optional[str] = None
        self.lock = threading.Lock()
        self.samples: List[Dict[str, str]] = []
        self.last_result: Optional[Dict] = None
        # 机械臂分拣单元：默认 auto，检测出 NG 自动入队剔除
        self.robot = RobotController(max_queue=8)

    def ensure_loaded(self) -> None:
        with self.lock:
            if self.inspector is not None or self.load_error:
                return
            try:
                self.inspector = DefectInspector(self.cfg)
                self.samples = collect_samples(self.cfg)
            except Exception as e:  # 模型缺失等
                self.load_error = f"{type(e).__name__}: {e}"

    def get_agent(self):
        """按需初始化智能体（不预先测试网络，避免启动变慢）。"""
        if self.agent is None:
            from agent.graph import create_agent
            self.agent = create_agent(self.cfg, enable_llm=self.cfg.agent.enable_llm, verbose=False)
        return self.agent


def collect_samples(cfg: AppConfig, per_class: int = 8) -> List[Dict[str, str]]:
    """从测试集每个类别随机抽几张，作为网页里的「示例图片」。"""
    rng = random.Random(2026)
    out: List[Dict[str, str]] = []
    test_dir = Path(cfg.test_dir)
    if not test_dir.exists():
        return out
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        files = list_images(cls_dir)
        if not files:
            continue
        for f in rng.sample(files, min(per_class, len(files))):
            out.append({"path": str(f), "label": cls_dir.name, "name": f.name})
    return out


STATE: Optional[WebState] = None


# ================================================================ 业务处理
def to_data_url(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def read_upload(stream, field: cgi.FieldStorage) -> Path:
    """把浏览器上传的图片落盘到 outputs/visualizations/web_uploads。"""
    WEB_UPLOADS.mkdir(parents=True, exist_ok=True)
    filename = Path(field.filename or "upload.png").name
    suffix = Path(filename).suffix.lower() or ".png"
    if suffix not in {e for e in IMG_EXTS}:
        suffix = ".png"
    data = field.file.read()
    if not data:
        raise ValueError("上传的文件是空的")
    target = WEB_UPLOADS / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    target.write_bytes(data)
    return target


def is_uploaded_path(path: str) -> bool:
    try:
        Path(path).resolve().relative_to(WEB_UPLOADS.resolve())
        return True
    except (ValueError, OSError):
        return False


def run_detection(state: WebState, image_path: Path, with_llm: bool) -> Dict:
    state.ensure_loaded()
    if state.inspector is None:
        raise RuntimeError(state.load_error or "检测器未能初始化")
    out_dir = VIS_DIR / "web"
    result = state.inspector.inspect_image(image_path, out_dir=out_dir, save_visual=True)
    # 检测结果直接进分拣单元：NG 入剔除队列，OK 放行
    sorting = state.robot.ingest(result)
    analysis = ""
    agent_note = ""
    if with_llm:
        try:
            agent = state.get_agent()
            final = agent.run(f"请质检这张图片并简要说明结论与处置建议：{image_path}")
            analysis = final.get("analysis", "")
            status = agent.status
            agent_note = (f"{status.provider}/{status.model}" if status.enabled
                          else status.reason)
        except Exception as e:
            agent_note = f"大模型解读失败，已改用规则模板：{type(e).__name__}: {e}"
    if not analysis:
        from agent.graph import template_analysis
        analysis = template_analysis([{**_public(result), "_tool": "inspect_image"}], [])

    base = REPORT_DIR
    frag = report_render.render_result_fragments(result, base)
    payload = {
        "ok": True,
        "fragments": frag,
        "analysis": analysis,
        "agent_note": agent_note,
        "prepared_image": to_data_url(result["_prepared"]),
        "sorting": sorting,
        "meta": {
            "image_name": result["image_name"],
            "image_path": result["image_path"],
            "verdict": result["verdict"],
            "severity": result["severity"],
            "anomaly_score": result["anomaly_score"],
            "threshold": result["threshold"],
            "threshold_source": result["threshold_source"],
            "defect_ratio": result["defect_ratio"],
            "num_regions": result["num_regions"],
            "inference_ms": result["inference_ms"],
            "recommendation": result["recommendation"],
            "verdict_reason": result.get("verdict_reason", ""),
        },
    }
    with state.lock:
        state.last_result = {**payload, "full": _public(result)}
    return payload


def _public(result: Dict) -> Dict:
    return {k: v for k, v in strip_internal(result).items()}


def export_report(state: WebState, analysis: str = "", fmt: str = "html") -> Dict[str, str]:
    """把「最近一次检测」导出成离线报告（html / markdown / docx），不重复推理。"""
    if state.last_result is None:
        raise RuntimeError("还没有检测结果，请先检测一张图片")
    meta = state.last_result["meta"]
    if state.inspector is None:
        raise RuntimeError("检测器不可用")
    fmt = str(fmt or "html").lower()
    if fmt not in ("html", "markdown", "docx"):
        raise ValueError("format 只支持 html / markdown / docx")
    # 结果里已带有 visuals；重新取一次公开字段即可（inspect_image 不重复落盘）
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"qc_web_{Path(meta['image_name']).stem}_{stamp}"

    # 最近一次检测的完整结构化结果（含 visuals）
    full = state.last_result.get("full") or {}
    if not full:
        raise RuntimeError("检测结果不完整，请重新检测")
    row = _bundle_visuals(strip_internal(full), REPORT_DIR)
    text = analysis or state.last_result.get("analysis", "")
    meta_info = {"line_name": state.cfg.report.line_name}
    md = report_render.render_markdown(row, meta=meta_info, analysis=text, base=REPORT_DIR)
    html_text = report_render.render_html(row, meta=meta_info, analysis=text, base=REPORT_DIR)
    md_path = REPORT_DIR / f"{file_name}.md"
    html_path = REPORT_DIR / f"{file_name}.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    paths = {"html": str(html_path), "markdown": str(md_path)}
    if fmt == "docx":
        docx_path = REPORT_DIR / f"{file_name}.docx"
        report_render.render_docx(row, meta=meta_info, analysis=text, base=REPORT_DIR,
                                  out_path=docx_path)
        paths["docx"] = str(docx_path)
    return paths


# ================================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "PatchCoreQC/1.0"

    def log_message(self, fmt, *args):  # 安静一些，只在出错时提示
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            sys.stderr.write("[web] %s\n" % (fmt % args))

    # ---------------------------------------------------------- 工具
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: Optional[Dict[str, str]] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _error(self, msg: str, code: int = 400):
        self._json({"ok": False, "error": msg}, code)

    def _serve_file(self, path: Path, ctype: str):
        if not path.exists() or not path.is_file():
            return self._error("文件不存在", 404)
        self._send(200, path.read_bytes(), ctype)

    # ---------------------------------------------------------- 路由
    def do_GET(self):
        from urllib.parse import urlparse, unquote
        url = urlparse(self.path)
        p = unquote(url.path)
        if p in ("/", "/index.html", "/_shot.html"):
            # /_shot.html 是给无头浏览器截图用的调试入口（与首页相同）
            return self._serve_file(WEB_DIR / ("index.html" if p != "/_shot.html" else "_shot.html"),
                                    "text/html; charset=utf-8")
        if p == "/api/status":
            STATE.ensure_loaded()
            return self._json({
                "ok": STATE.inspector is not None,
                "error": STATE.load_error,
                "model": STATE.inspector.model.backbone if STATE.inspector else None,
                "device": STATE.inspector.model.device if STATE.inspector else None,
                "threshold": round(STATE.inspector.thresholds.image_score, 5) if STATE.inspector else None,
                "threshold_source": STATE.inspector.thresholds.source if STATE.inspector else None,
                "samples": STATE.samples,
                "llm": _llm_hint(STATE.cfg),
            })
        if p.startswith("/files/"):
            raw = p[len("/files/"):].lstrip("/")
            target = Path(raw)
            if not target.is_absolute():
                target = PROJECT_ROOT / raw
            target = target.resolve()
            if not target.is_file():
                return self._error("文件不存在", 404)
            suffix = target.suffix.lower()
            ctype = ("image/png" if suffix == ".png" else
                     "image/jpeg" if suffix in (".jpg", ".jpeg") else
                     "text/html; charset=utf-8" if suffix == ".html" else
                     "text/markdown; charset=utf-8" if suffix == ".md" else
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                     if suffix == ".docx" else
                     "image/svg+xml" if suffix == ".svg" else
                     "application/octet-stream")
            return self._serve_file(target, ctype)
        if p == "/api/random":
            STATE.ensure_loaded()
            if not STATE.samples:
                return self._error("测试集中没有可用的示例图片", 404)
            return self._json({"ok": True, "sample": random.choice(STATE.samples)})
        if p == "/api/robot/status":
            return self._json(STATE.robot.status())
        return self._error("未知路径", 404)

    def do_POST(self):
        from urllib.parse import urlparse
        p = urlparse(self.path).path
        if p == "/api/detect":
            return self._handle_detect()
        if p == "/api/detect_path":
            return self._handle_detect_path()
        if p == "/api/export":
            return self._handle_export()
        if p == "/api/fragments":
            return self._handle_fragments()
        if p == "/api/robot/mode":
            return self._handle_robot_mode()
        if p == "/api/robot/estop":
            STATE.robot.estop()
            return self._json(STATE.robot.status())
        if p == "/api/robot/reset":
            STATE.robot.reset()
            return self._json(STATE.robot.status())
        if p == "/api/robot/fault":
            return self._handle_robot_fault()
        return self._error("未知接口", 404)

    # ---------------------------------------------------------- 接口
    def _handle_detect(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._error("请以 multipart/form-data 上传图片")
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                    environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype})
            field = form["image"] if "image" in form else None
            if field is None or not getattr(field, "file", None):
                return self._error("没有收到图片文件")
            path = read_upload(self.rfile, field)
            with_llm = str(form.getvalue("with_llm", "0")) in ("1", "true", "on")
            payload = run_detection(STATE, path, with_llm)
            payload["analysis"] = payload["analysis"]
            return self._json(payload)
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)

    def _handle_detect_path(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            raw = str(data.get("path", "")).strip().strip('"')
            if not raw:
                return self._error("缺少图片路径")
            path = Path(raw)
            if not path.is_absolute():
                path = (PROJECT_ROOT / raw)
            if not path.exists():
                return self._error(f"图片不存在：{path}")
            payload = run_detection(STATE, path, bool(data.get("with_llm")))
            return self._json(payload)
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)

    def _handle_fragments(self):
        """只返回可视化 HTML 片段（图片走 /files/ URL），供页面刷新后复用。"""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            full = data.get("full") or {}
            if not full:
                return self._error("缺少检测结果")
            return self._json({"ok": True, "fragments": report_render.render_result_fragments(full, REPORT_DIR)})
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)

    def _handle_export(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            fmt = str(data.get("format", "html")).lower()
            paths = export_report(STATE, str(data.get("analysis", "")), fmt=fmt)
            target = paths.get(fmt)
            if not target:
                return self._error(f"不支持的导出格式：{fmt}", 400)
            return self._json({"ok": True, "format": fmt, **paths,
                               "url": "/files/" + Path(target).as_posix()})
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)

    def _handle_robot_mode(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            mode = STATE.robot.set_mode(str(data.get("mode", "auto")))
            return self._json({"ok": True, "mode": mode, "robot": STATE.robot.status()})
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}", 400)

    def _handle_robot_fault(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            name = data.get("type") or None
            if name not in (None, "", "vacuum_loss", "jam"):
                return self._error("fault 只支持 vacuum_loss / jam / 空（清除）", 400)
            fault = STATE.robot.set_fault(name)
            return self._json({"ok": True, "fault": fault or "", "robot": STATE.robot.status()})
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}", 400)


def _llm_hint(cfg: AppConfig) -> Dict:
    from agent.llm import resolve_provider
    provider, base_url, model, api_key = resolve_provider(cfg.agent)
    return {"enabled": bool(api_key) and provider not in (None, "none"),
            "provider": provider, "model": model}


# ================================================================ 启动
def find_free_port(host: str, start: int, tries: int = 30) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    return start


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="PatchCore 质检网页版")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--config", default=None)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--no-llm", action="store_true", help="网页里禁用大模型解读")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.no_llm:
        cfg.agent.enable_llm = False
    global STATE
    STATE = WebState(cfg)
    STATE.robot.start()
    print("[web] 机械臂分拣单元已启动（仿真对象，auto 模式）", flush=True)
    print("[web] 正在加载 PatchCore 模型 ...", flush=True)
    STATE.ensure_loaded()
    if STATE.inspector is None:
        print(f"[web] 模型加载失败：{STATE.load_error}")
        print("[web] 请先运行: python scripts/train.py")
    else:
        m = STATE.inspector.model
        print(f"[web] 模型就绪：{m.backbone} / 设备 {m.device} / 阈值 "
              f"{STATE.inspector.thresholds.image_score:.4f}（{STATE.inspector.thresholds.source}）")
        print(f"[web] 示例图片 {len(STATE.samples)} 张，来自测试集各类别")

    port = find_free_port(args.host, args.port)
    url = f"http://{args.host}:{port}/"
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    print(f"[web] 打开浏览器访问: {url}")
    print("[web] 按 Ctrl+C 停止服务")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
