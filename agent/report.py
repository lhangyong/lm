"""报告渲染：把结构化质检结果渲染成 Markdown / HTML（离线、无外网字体依赖）。

HTML 里内嵌 SVG（热力图、分数直方图）以及 <img> 引用的叠加图；
图片一律用相对路径，便于整个 outputs/reports 目录拷走直接打开。
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

from patchcore_qc.detector import DEFECT_SEMANTICS


def _rel(target: Optional[str], base: Path) -> Optional[str]:
    """报告目录内的文件用相对路径（整目录可拷走）；目录外的文件在拷贝失败时退回 file:// 绝对路径。"""
    if not target:
        return None
    p = Path(target).resolve()
    if not p.exists():
        return None
    try:
        return p.relative_to(base.resolve()).as_posix()
    except ValueError:
        return p.as_uri()


def verdict_badge(verdict: str) -> str:
    color = "#e5484d" if verdict == "NG" else "#30a46c"
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
            f'background:{color};color:#fff;font-weight:600;font-size:13px">{verdict}</span>')


def severity_badge(sev: str) -> str:
    colors = {"Major": "#e5484d", "Minor": "#f5a524", "Trace": "#0090ff", "None": "#6b7280"}
    return (f'<span style="display:inline-block;padding:1px 8px;border-radius:8px;'
            f'background:{colors.get(sev, "#6b7280")};color:#fff;font-size:12px">{sev}</span>')


def _url(path: Optional[str]) -> Optional[str]:
    """文件路径 -> 适合放进网页的 /files/ URL（逐段编码，避开中文与空格问题）。"""
    if not path:
        return None
    return "/files/" + "/".join(quote(part) for part in Path(path).as_posix().split("/"))


def render_result_fragments(item: Dict, base: Path) -> Dict[str, str]:
    """返回可直接拼进网页的 HTML 片段：可视化图 + 缺陷区域表。

    这是给本地 Web 界面（agent/web.py）用的：
    - 图片通过 /files/<绝对路径> 由本地服务读取，不依赖相对路径；
    - 热力图、直方图以 SVG 内联，避免额外请求。
    """
    visuals = item.get("visuals") or {}
    overlay_url = _url(visuals.get("overlay_png"))
    overlay = ""
    if overlay_url:
        cap = ("异常叠加（白边为缺陷轮廓，彩色框为检出区域）" if item.get("regions")
               else "异常叠加（未检出缺陷区域）")
        overlay = (f'<figure><img src="{html.escape(overlay_url)}" alt="overlay"/>'
                   f'<figcaption>{cap}</figcaption></figure>')
    svgs = []
    for key, cap in (("heatmap_svg", "PatchCore 热力图"), ("hist_svg", "patch 异常分数分布")):
        p = visuals.get(key)
        if p and Path(p).exists():
            svgs.append(f'<figure class="svg">{Path(p).read_text(encoding="utf-8")}'
                        f'<figcaption>{cap}</figcaption></figure>')
    rows = "".join(
        f"<tr><td>#{i + 1}</td><td>{html.escape(r.get('location_detail', r['location']))}</td>"
        f"<td>{r['area_ratio']:.4%}</td><td>{r.get('significance', 0):.2f}</td>"
        f"<td>{severity_badge(r['severity'])}</td></tr>"
        for i, r in enumerate(item.get("regions", [])))
    table = (f'<table><thead><tr><th>#</th><th>位置</th><th>面积占比</th><th>显著性</th><th>级别</th></tr>'
             f'</thead><tbody>{rows}</tbody></table>') if rows else ""
    return {
        "overlay": overlay,
        "svgs": "".join(svgs),
        "regions": table,
        "summary": f'{verdict_badge(item.get("verdict", "OK"))} '
                   f'{severity_badge(item.get("severity", "None"))} · '
                   f'分数 {item.get("anomaly_score", 0):.4f} / 阈值 {item.get("threshold", 0):.4f}',
    }


def render_hero(item: Dict, base: Path) -> str:
    """单图结果卡片：原图 / 叠加图 / 热力图 / 直方图 + 指标表。"""
    visuals = item.get("visuals") or {}
    imgs = []
    src = item.get("image_path")
    if src and Path(src).exists():
        src_rel = _rel(src, base)
        if src_rel:
            imgs.append(f'<figure><img src="{html.escape(src_rel)}" alt="source"/>'
                        f'<figcaption>原始图像</figcaption></figure>')
    for key, cap in (("overlay_png", "异常叠加（越大越可疑）"),
                     ("heatmap_svg", "PatchCore 热力图"), ("hist_svg", "patch 异常分数分布")):
        p = visuals.get(key)
        if not p:
            continue
        if key.endswith("_svg") and Path(p).exists():
            imgs.append(f'<figure class="svg">{Path(p).read_text(encoding="utf-8")}'
                        f'<figcaption>{cap}</figcaption></figure>')
        else:
            rel = _rel(p, base)
            if rel:
                imgs.append(f'<figure><img src="{html.escape(rel)}" alt="{key}"/>'
                            f'<figcaption>{cap}</figcaption></figure>')
    regions_rows = "".join(
        f"<tr><td>#{i + 1}</td><td>{html.escape(r.get('location_detail', r['location']))}</td>"
        f"<td>{r['area_ratio']:.4%}</td><td>{r.get('peak_ratio', 1):.2f}×</td>"
        f"<td>{severity_badge(r['severity'])}</td></tr>"
        for i, r in enumerate(item.get("regions", [])))
    table = ""
    if regions_rows:
        table = ("<table><thead><tr><th>#</th><th>位置</th><th>面积占比</th><th>显著度</th><th>级别</th></tr>"
                 f"</thead><tbody>{regions_rows}</tbody></table>")
    return f"""
<section class="card">
  <h3>{html.escape(item['image_name'])} &nbsp;{verdict_badge(item['verdict'])} &nbsp;{severity_badge(item.get('severity', 'None'))}</h3>
  <div class="metrics">
    <div><b>异常分数</b><br>{item['anomaly_score']:.4f}</div>
    <div><b>判定阈值</b><br>{item['threshold']:.4f}<br><small>{item.get('threshold_source', '')}</small></div>
    <div><b>缺陷区域</b><br>{item.get('num_regions', 0)} 处</div>
    <div><b>异常面积</b><br>{item.get('defect_ratio', 0):.4%}</div>
    <div><b>处理建议</b><br>{html.escape(item.get('recommendation', ''))}</div>
  </div>
  <div class="gallery">{''.join(imgs)}</div>
  {table}
</section>"""


def render_markdown(item: Dict, meta: Optional[Dict] = None, analysis: Optional[str] = None,
                    base: Optional[Path] = None) -> str:
    meta = meta or {}
    lines = [f"# 质检报告：{item['image_name']}", ""]
    if meta.get("line_name"):
        lines.append(f"- 产线：{meta['line_name']}")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 检测模型：PatchCore（{item['model'].get('backbone', '-')}，"
                 f"memory bank {item['model'].get('coreset', 0)} patches）")
    lines.append(f"- 判定结果：**{item['verdict']}**（严重度 {item.get('severity', 'None')}）")
    lines.append(f"- 异常分数：{item['anomaly_score']:.4f}（阈值 {item['threshold']:.4f}，"
                 f"来源 {item.get('threshold_source')}）")
    lines.append(f"- 缺陷区域：{item.get('num_regions', 0)} 处，异常面积占比 {item.get('defect_ratio', 0):.4%}")
    if item.get("verdict") == "OK":
        lines.append("- 说明：未检出超过阈值的局部显著异常，无缺陷区域可标注")
    if item.get("verdict_reason"):
        lines.append(f"- 判定依据：{item['verdict_reason']}")
    lines.append("")
    if item.get("regions"):
        lines += ["| # | 位置 | 面积占比 | 显著度 | 级别 |", "|---|---|---|---|---|"]
        for i, r in enumerate(item["regions"]):
            lines.append(f"| {i + 1} | {r.get('location_detail', r['location'])} | {r['area_ratio']:.4%} | "
                         f"{r.get('significance', 0):.2f} | {r['severity']} |")
        lines.append("")
    if analysis:
        lines += ["## 智能体分析", "", analysis.strip(), ""]
    visuals = item.get("visuals") or {}
    if visuals and base is not None:
        lines.append("## 可视化")
        for key, cap in (("overlay_png", "叠加图"), ("heatmap_svg", "热力图"), ("hist_svg", "分数分布")):
            rel = _rel(visuals.get(key), base)
            if rel:
                lines.append(f"- [{cap}]({rel})")
        lines.append("")
    lines += ["## 处理建议", "", f"{item.get('recommendation', '')}", ""]
    return "\n".join(lines)


HTML_STYLE = """
:root{color-scheme:dark light}
body{margin:0;padding:32px;font-family:'Segoe UI','Microsoft YaHei',system-ui,sans-serif;
     background:#0f1115;color:#e6e6e6;line-height:1.6}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}
.sub{color:#9aa0a6;margin-bottom:24px}
.card{background:#171a21;border:1px solid #262b36;border-radius:14px;padding:20px 22px;margin:18px 0}
.metrics{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.metrics>div{flex:1 1 140px;background:#11141a;border-radius:10px;padding:10px 12px;font-size:14px}
.metrics b{color:#9aa0a6;font-weight:500;font-size:12px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:14px 0}
figure{margin:0}
figure img,figure svg{width:100%;border-radius:10px;display:block;background:#0b0d11}
figcaption{color:#9aa0a6;font-size:12px;margin-top:6px;text-align:center}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}
th,td{padding:7px 10px;border-bottom:1px solid #262b36;text-align:left}
th{color:#9aa0a6;font-weight:500}
.kv{display:grid;grid-template-columns:160px 1fr;gap:6px 14px;font-size:14px}
.kv div:nth-child(odd){color:#9aa0a6}
pre{white-space:pre-wrap;background:#11141a;border-radius:10px;padding:14px;font-size:13.5px;
    font-family:'Cascadia Mono',Consolas,monospace;line-height:1.65}
.banner{font-size:15px;padding:12px 16px;border-radius:10px}
"""


def render_html(item: Dict, meta: Optional[Dict] = None, analysis: Optional[str] = None,
                base: Optional[Path] = None, extra_sections: str = "") -> str:
    base = base or Path(".")
    meta = meta or {}
    ok = item["verdict"] == "OK"
    banner_bg = "#12301f" if ok else "#3a1518"
    banner = (f'<div class="banner" style="background:{banner_bg}">'
              f'{"✅ 判定为 OK（合格），可放行" if ok else "⛔ 判定为 NG（不合格），需拦截处理"}'
              f' —— 异常分数 {item["anomaly_score"]:.4f} / 阈值 {item["threshold"]:.4f}</div>')
    kv = "".join(f"<div>{html.escape(str(k))}</div><div>{v}</div>" for k, v in [
        ("产线 / 工位", html.escape(meta.get("line_name", "-"))),
        ("报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("检测模型", f"PatchCore · {html.escape(str(item['model'].get('backbone')))} · "
                     f"memory bank {item['model'].get('coreset', 0)} patches"),
        ("阈值来源", html.escape(str(item.get("threshold_source", "-")))),
        ("判定逻辑", "异常分数 ≥ 图像阈值，且存在局部显著缺陷区域 → NG；"
                     "区域面积 ≥ 10% 为 Major、≥ 2% 为 Minor，其余为 Trace"),
    ])
    analysis_html = ""
    if analysis:
        analysis_html = f'<section class="card"><h3>智能体分析</h3><pre>{html.escape(analysis.strip())}</pre></section>'
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>质检报告 · {html.escape(item['image_name'])}</title>
<style>{HTML_STYLE}</style></head>
<body><div class="wrap">
<h1>工业视觉质检报告</h1>
<div class="sub">{html.escape(item['image_name'])} · PatchCore 无监督缺陷检测 · LangGraph 智能体生成</div>
{banner}
<section class="card"><h3>基本信息</h3><div class="kv">{kv}</div></section>
{render_hero(item, base)}
{analysis_html}
{extra_sections}
<section class="card"><h3>说明</h3>
<p style="color:#9aa0a6;font-size:13px">本报告由「PatchCore 无监督缺陷检测 + LangGraph 质检智能体」自动生成。
检测模型仅使用正常样本训练，不使用任何缺陷标注；缺陷判定阈值由训练集正常样本分数自动标定，
判定结果需结合产线工艺规范复核。</p></section>
</div></body></html>"""


def render_batch_markdown(rows: List[Dict], meta: Optional[Dict] = None, analysis: Optional[str] = None) -> str:
    meta = meta or {}
    ng = [r for r in rows if r["verdict"] == "NG"]
    lines = ["# 批量质检汇总报告", "",
             f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 产线 / 工位：{meta.get('line_name', '-')}",
             f"- 检测数量：{len(rows)}，NG {len(ng)}，OK {len(rows) - len(ng)}，"
             f"不良率 {len(ng) / max(1, len(rows)):.1%}",
             ""]
    if analysis:
        lines += ["## 智能体分析", "", analysis.strip(), ""]
    lines += ["| 序号 | 图片 | 判定 | 严重度 | 异常分数 | 阈值 | 区域数 | 面积占比 |",
              "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        lines.append(f"| {i + 1} | {r['image_name']} | {r['verdict']} | {r.get('severity', 'None')} | "
                     f"{r['anomaly_score']:.4f} | {r['threshold']:.4f} | {r.get('num_regions', 0)} | "
                     f"{r.get('defect_ratio', 0):.4%} |")
    lines.append("")
    if ng:
        lines += ["## NG 明细", ""]
        for r in ng[:50]:
            locs = "、".join(reg.get("location_detail", reg["location"]) for reg in r.get("regions", [])[:3]) or "-"
            lines.append(f"- **{r['image_name']}**（{r['severity']}，分数 {r['anomaly_score']:.4f}）："
                         f"位置 {locs}；{r.get('recommendation', '')}")
    return "\n".join(lines)


def render_docx(item: Dict, meta: Optional[Dict] = None, analysis: Optional[str] = None,
                base: Optional[Path] = None, out_path: Optional[Path] = None) -> Path:
    """把单图结果导出成 Word（.docx），图片用报告 assets 目录里的文件。"""
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    if out_path is None:
        raise ValueError("out_path 必填")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")

    def _run(run, size=10.5, bold=False, color=None, font="宋体"):
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.bold = bold
        if color:
            run.font.color.rgb = RGBColor(*color)
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font)

    def heading(text, level=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        p.paragraph_format.space_after = Pt(6 if level == 1 else 4)
        _run(p.add_run(text), 15 if level == 1 else 12, True, font="黑体")
        return p

    def body(text, size=10.5, bold=False, color=None, align=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        if align is not None:
            p.alignment = align
        _run(p.add_run(str(text)), size, bold, color)
        return p

    def cell(c, text, bold=False, size=9.5, align=None):
        c.text = ""
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        if align is not None:
            p.alignment = align
        _run(p.add_run(str(text)), size, bold)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    _run(title.add_run("工业视觉质检报告"), 18, True, font="黑体")

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(14)
    _run(sub.add_run(f"{item['image_name']} · PatchCore 无监督缺陷检测 · LangGraph 智能体生成"),
         9.5, False, (0x66, 0x66, 0x66))

    ok = item.get("verdict") == "OK"
    verdict = doc.add_paragraph()
    verdict.alignment = WD_ALIGN_PARAGRAPH.CENTER
    verdict.paragraph_format.space_after = Pt(8)
    _run(verdict.add_run("判定为 OK（合格），可放行" if ok else "判定为 NG（不合格），需拦截处理"),
         13, True, (0x18, 0x7A, 0x3F) if ok else (0xC0, 0x28, 0x2D))

    info = [
        ("产线 / 工位", meta.get("line_name", "-")),
        ("报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("检测模型", f"PatchCore · {item['model'].get('backbone', '-')} · "
                     f"memory bank {item['model'].get('coreset', 0)} patches"),
        ("阈值来源", item.get("threshold_source", "-")),
        ("判定逻辑", "异常分数 ≥ 图像阈值，且存在局部显著缺陷区域 → NG；"
                     "区域面积 ≥ 10% 为 Major、≥ 2% 为 Minor，其余为 Trace"),
    ]
    table = doc.add_table(rows=len(info), cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(info):
        cell(table.rows[i].cells[0], k, True, 9.5)
        cell(table.rows[i].cells[1], v, False, 9.5)

    heading("检测指标", 2)
    metrics = [
        ("异常分数", f"{item['anomaly_score']:.4f}"),
        ("判定阈值", f"{item['threshold']:.4f}"),
        ("缺陷区域", f"{item.get('num_regions', 0)} 处"),
        ("异常面积", f"{item.get('defect_ratio', 0):.4%}"),
        ("严重度", item.get("severity", "None")),
        ("处理建议", item.get("recommendation", "")),
    ]
    mt = doc.add_table(rows=3, cols=4)
    mt.style = "Table Grid"
    mt.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (k, v) in enumerate(metrics):
        r, c = divmod(idx, 2)
        cell(mt.rows[r].cells[c * 2], k, True, 9.5)
        cell(mt.rows[r].cells[c * 2 + 1], v, False, 9.5)

    if item.get("verdict_reason"):
        body("判定依据：" + str(item["verdict_reason"]), 9.5)

    rows = item.get("regions") or []
    if rows:
        heading("缺陷区域", 2)
        rt = doc.add_table(rows=1, cols=5)
        rt.style = "Table Grid"
        rt.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c, h in enumerate(("#", "位置", "面积占比", "显著度", "级别")):
            cell(rt.rows[0].cells[c], h, True, 9.0, WD_ALIGN_PARAGRAPH.CENTER)
        for i, r in enumerate(rows):
            tr = rt.add_row()
            vals = (str(i + 1), r.get("location_detail", r.get("location", "")),
                    f"{r.get('area_ratio', 0):.4%}", f"{r.get('significance', 0):.2f}",
                    str(r.get("severity", "Trace")))
            for c, v in enumerate(vals):
                cell(tr.cells[c], v, False, 9.0,
                     WD_ALIGN_PARAGRAPH.CENTER if c in (0, 2, 3, 4) else None)

    heading("可视化", 2)
    figs = []
    src = item.get("image_path")
    if src and Path(src).exists():
        figs.append(("原始图像", Path(src)))
    overlay = (item.get("visuals") or {}).get("overlay_png")
    if overlay and Path(overlay).exists():
        figs.append(("异常叠加图", Path(overlay)))
    for cap, path in figs:
        try:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(path), width=Cm(13.5))
            c = doc.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _run(c.add_run(cap), 9, False, (0x66, 0x66, 0x66))
        except Exception as exc:
            body(f"{cap}插入失败：{exc}", 9)

    if analysis:
        heading("智能体分析", 2)
        lines = [ln.rstrip() for ln in str(analysis).splitlines()]
        for ln in lines:
            s = ln.strip()
            if not s:
                doc.add_paragraph()
                continue
            if s.startswith("#"):
                heading(s.lstrip("#").strip(), 2)
                continue
            if s.startswith(("- ", "* ")):
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(2)
                _run(p.add_run(s[2:].replace("**", "").replace("`", "")), 10)
                continue
            body(s.replace("**", "").replace("`", ""), 10)

    heading("说明", 2)
    note = ("本报告由「PatchCore 无监督缺陷检测 + LangGraph 质检智能体」自动生成。"
            "检测模型仅使用正常样本训练，不使用任何缺陷标注；阈值由训练集正常样本分数自动标定，"
            "判定结果需结合产线工艺规范复核。")
    body(note, 9, False, (0x66, 0x66, 0x66))

    doc.save(str(out_path))
    return out_path


def render_batch_html(rows: List[Dict], meta: Optional[Dict] = None, analysis: Optional[str] = None,
                      base: Optional[Path] = None) -> str:
    base = base or Path(".")
    meta = meta or {}
    ng = [r for r in rows if r["verdict"] == "NG"]
    rate = len(ng) / max(1, len(rows))
    color = "#e5484d" if ng else "#30a46c"
    trs = "".join(
        f"<tr><td>{i + 1}</td><td>{html.escape(r['image_name'])}</td><td>{verdict_badge(r['verdict'])}</td>"
        f"<td>{severity_badge(r.get('severity', 'None'))}</td><td>{r['anomaly_score']:.4f}</td>"
        f"<td>{r['threshold']:.4f}</td><td>{r.get('num_regions', 0)}</td>"
        f"<td>{r.get('defect_ratio', 0):.4%}</td></tr>" for i, r in enumerate(rows))
    figs = []
    for r in ng[:6]:
        p = _rel((r.get("visuals") or {}).get("overlay_png"), base)
        if p:
            figs.append(f'<figure><img src="{html.escape(p)}" alt="{html.escape(r["image_name"])}"/>'
                        f'<figcaption>{html.escape(r["image_name"])} · {r.get("severity")}</figcaption></figure>')
    gallery = f'<section class="card"><h3>NG 样例</h3><div class="gallery">{"".join(figs)}</div></section>' if figs else ""
    analysis_html = f'<section class="card"><h3>智能体分析</h3><pre>{html.escape(analysis.strip())}</pre></section>' if analysis else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>批量质检汇总报告</title><style>{HTML_STYLE}</style></head><body><div class="wrap">
<h1>批量质检汇总报告</h1>
<div class="sub">{html.escape(meta.get('line_name', ''))} · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
<div class="banner" style="background:#11141a">检测 <b>{len(rows)}</b> 件 ·
<span style="color:{color};font-weight:600">NG {len(ng)} 件（不良率 {rate:.1%}）</span> · OK {len(rows) - len(ng)} 件</div>
{analysis_html}
<section class="card"><h3>明细</h3>
<table><thead><tr><th>#</th><th>图片</th><th>判定</th><th>严重度</th><th>异常分数</th><th>阈值</th>
<th>区域数</th><th>面积占比</th></tr></thead><tbody>{trs}</tbody></table></section>
{gallery}
</div></body></html>"""


def render_general_html(text: str, meta: Optional[Dict] = None) -> str:
    meta = meta or {}
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>质检助手回答</title><style>{HTML_STYLE}</style></head><body><div class="wrap">
<h1>质检助手回答</h1>
<div class="sub">{html.escape(meta.get('line_name', ''))} · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
<section class="card"><pre>{html.escape(text.strip())}</pre></section>
</div></body></html>"""
