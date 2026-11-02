"""LLM 适配层：选择 provider、构造聊天模型、处理无 Key/无网络/不支持 tool calling 的情况。

支持 OpenAI 兼容端点：DeepSeek / OpenAI / 通义千问（DashScope 兼容模式）/ Ollama / 自建服务。
环境变量：
    DEEPSEEK_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY / QC_AGENT_API_KEY
    QC_AGENT_PROVIDER=deepseek|openai|qwen|ollama|none
    QC_AGENT_MODEL=...        QC_AGENT_BASE_URL=...
"""
from __future__ import annotations

import base64
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

from patchcore_qc.config import AgentConfig

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY",
                 "tools": True},
    "openai": {"base_url": None, "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY", "tools": True},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus",
             "key_env": "DASHSCOPE_API_KEY", "tools": True},
    "ollama": {"base_url": "http://127.0.0.1:11434/v1", "model": "qwen2.5:7b", "key_env": None, "tools": True},
}
# 已知不支持 function calling 的模型前缀（走 JSON 计划模式）
NO_TOOLS_PREFIXES = ("deepseek-r1", "deepseek-reasoner", "o1", "qwq")


@dataclass
class LLMStatus:
    enabled: bool = False
    provider: str = "none"
    model: str = "-"
    base_url: Optional[str] = None
    reason: str = ""
    tools: bool = False
    error: Optional[str] = None


@dataclass
class LLMBundle:
    llm: Any = None
    status: LLMStatus = None


def resolve_provider(cfg: AgentConfig) -> tuple[str, Optional[str], str, Optional[str]]:
    provider = (cfg.provider or "auto").lower()
    if provider == "auto":
        provider = "none"
        for name, spec in PROVIDERS.items():
            env = spec.get("key_env")
            if env and os.environ.get(env):
                provider = name
                break
        if provider == "none" and os.environ.get("QC_AGENT_API_KEY") and os.environ.get("QC_AGENT_BASE_URL"):
            provider = "openai"
    spec = PROVIDERS.get(provider, {})
    base_url = cfg.base_url or os.environ.get("QC_AGENT_BASE_URL") or spec.get("base_url")
    model = cfg.model or os.environ.get("QC_AGENT_MODEL") or spec.get("model")
    api_key = cfg.api_key or os.environ.get("QC_AGENT_API_KEY")
    if not api_key and spec.get("key_env"):
        api_key = os.environ.get(spec["key_env"])
    if provider == "ollama" and not api_key:
        api_key = "ollama"
    return provider, base_url, model or "-", api_key


def build_llm(cfg: AgentConfig, tools: Optional[List[Any]] = None) -> LLMBundle:
    """构造聊天模型；任何失败都不抛异常，返回 status 让上层降级为离线规则引擎。"""
    provider, base_url, model, api_key = resolve_provider(cfg)
    if not cfg.enable_llm:
        return LLMBundle(None, LLMStatus(False, provider, model, base_url, "配置里关闭了大模型（enable_llm=false）"))
    if provider == "none" or not api_key:
        return LLMBundle(None, LLMStatus(False, provider, model, base_url,
                                         "未找到 API Key（设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 即可接入大模型）；"
                                         "当前使用离线规则引擎出报告"))
    try:
        from langchain_openai import ChatOpenAI
    except Exception as e:
        return LLMBundle(None, LLMStatus(False, provider, model, base_url, "未安装 langchain-openai", error=str(e)))

    kwargs = {"model": model, "temperature": cfg.temperature, "api_key": api_key,
              "max_retries": 1, "timeout": 90}
    if base_url:
        kwargs["base_url"] = base_url
    llm = _try_init(ChatOpenAI, kwargs)
    if llm is None:
        return LLMBundle(None, LLMStatus(False, provider, model, base_url,
                                         "ChatOpenAI 初始化失败", error="参数组合均不可用"))

    status = LLMStatus(True, provider, model, base_url, "ok")
    if tools and not model.lower().startswith(NO_TOOLS_PREFIXES):
        try:
            bound = llm.bind_tools(tools)
            bound.invoke  # noqa: B018 - 仅确认返回对象可用
            status.tools = True
            return LLMBundle(bound, status)
        except Exception as e:
            status.error = str(e)[:200]
    elif tools:
        status.error = "该模型不支持 function calling，改用 JSON 计划模式"
    return LLMBundle(llm, status)


def _try_init(cls, kwargs: Dict[str, Any]):
    variants = [kwargs]
    legacy = {k: v for k, v in kwargs.items() if k not in ("api_key", "base_url")}
    if "api_key" in kwargs:
        legacy["openai_api_key"] = kwargs["api_key"]
    if kwargs.get("base_url"):
        legacy["openai_api_base"] = kwargs["base_url"]
    variants.append(legacy)
    for kw in variants:
        try:
            return cls(**kw)
        except Exception:
            continue
    return None


def image_to_data_url(path: str | Path, max_side: int = 1024) -> Optional[str]:
    """图片转 data URL（长边压缩），用于多模态模型看图。"""
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None
