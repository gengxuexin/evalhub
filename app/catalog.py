"""可接入的 Agent 目录。

内置消防项目里的四个服务，也保存用户登记的 HTTP JSON Agent。
令牌只记录环境变量名，不写入目录文件。
"""

import json
import os
import re
from pathlib import Path

import httpx

from app.adapters.base import AgentAdapter
from app.adapters.fire_agent import FireAgentAdapter
from app.adapters.http_json import HttpJsonAdapter

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")
_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ENV = re.compile(r"^[A-Z][A-Z0-9_]{0,60}$")
_HEADER = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,40}$")

BUILTIN_AGENTS = (
    {
        "name": "fire_agent",
        "title": "法规问答",
        "adapter": "fire_agent",
        "kind": "文字",
        "usable_with_fire_qa": True,
        "base_url": "http://127.0.0.1:8001",
        "health_url": "http://127.0.0.1:8001/health",
        "endpoint": "POST /api/rag/ask",
        "summary": "根据消防法规回答问题。这套文字题可以直接评测。",
        "command": "python -m app.cli run --agent fire_agent --limit 3",
    },
    {
        "name": "fire_gateway",
        "title": "统一入口",
        "adapter": "http_json",
        "kind": "文字",
        "usable_with_fire_qa": True,
        "url": "http://127.0.0.1:8000/dispatch",
        "health_url": "http://127.0.0.1:8000/",
        "query_field": "query",
        "answer_field": "result.answer",
        "contexts_field": "result.contexts",
        "success_field": "status",
        "success_value": "success",
        "token_env": "GATEWAY_ACCESS_TOKEN",
        "token_header": "Authorization",
        "token_prefix": "Bearer ",
        "extra_fields": {"force_kind": "rag"},
        "top_k": None,
        "endpoint": "POST /dispatch",
        "summary": "接收问题后交给合适的能力处理。文字题会交给法规问答。",
        "command": "python -m app.cli run --agent fire_gateway --limit 3",
    },
    {
        "name": "agent2_voice",
        "title": "语音交互",
        "adapter": "unsupported",
        "kind": "语音",
        "usable_with_fire_qa": False,
        "health_url": "http://127.0.0.1:8002/health",
        "endpoint": "POST /api/voice/qa",
        "summary": "听录音并回答。这套题是文字问答，暂时不能评测语音。",
        "command": "",
    },
    {
        "name": "agent3_vision",
        "title": "图像检查",
        "adapter": "unsupported",
        "kind": "图像",
        "usable_with_fire_qa": False,
        "health_url": "http://127.0.0.1:8003/health",
        "endpoint": "POST /api/vision/analyze_image",
        "summary": "查看图片或视频中的消防隐患。这套题是文字问答，暂时不能评测图像。",
        "command": "",
    },
)

BUILTIN_NAMES = {item["name"] for item in BUILTIN_AGENTS}


def list_agents(directory: Path) -> list[dict]:
    saved = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(item, dict) and item.get("name") and item["name"] not in BUILTIN_NAMES:
                saved.append(item)
    return [*BUILTIN_AGENTS, *saved]


def get_agent(directory: Path, name: str) -> dict:
    for item in list_agents(directory):
        if item.get("name") == name:
            return item
    raise FileNotFoundError(f"没有这个 Agent：{name}")


def save_custom_agent(directory: Path, fields: dict) -> dict:
    name = str(fields.get("name") or "").strip()
    if not _NAME.fullmatch(name):
        raise ValueError("名称请用字母开头，只含字母、数字、下划线或短横线")
    if name in BUILTIN_NAMES:
        raise ValueError("这个名称留给消防项目里的 Agent")
    url = str(fields.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("地址需要以 http:// 或 https:// 开头")
    token_env = str(fields.get("token_env") or "").strip()
    if token_env and not _ENV.fullmatch(token_env):
        raise ValueError("令牌请填写环境变量名，不要填写令牌本身")
    token_header = str(fields.get("token_header") or "").strip()
    if token_header and not _HEADER.fullmatch(token_header):
        raise ValueError("认证头名称不正确")
    token_prefix = str(fields.get("token_prefix") or "")
    if token_prefix == "Bearer":
        token_prefix = "Bearer "
    if token_prefix not in {"", "Bearer "}:
        raise ValueError("认证前缀只支持空或 Bearer")
    health_url = str(fields.get("health_url") or "").strip()
    if health_url and not health_url.startswith(("http://", "https://")):
        raise ValueError("健康检查地址需要以 http:// 或 https:// 开头")
    profile = {
        "name": name,
        "title": str(fields.get("title") or name).strip() or name,
        "adapter": "http_json",
        "kind": "文字",
        "usable_with_fire_qa": True,
        "url": url,
        "health_url": health_url,
        "query_field": _field(fields.get("query_field"), "question"),
        "answer_field": _field(fields.get("answer_field"), "answer"),
        "contexts_field": _field(fields.get("contexts_field"), "contexts"),
        "success_field": _field(fields.get("success_field"), "status"),
        "success_value": _short(fields.get("success_value"), "success"),
        "token_env": token_env,
        "token_header": token_header,
        "token_prefix": token_prefix,
        "top_k": _top_k(fields.get("top_k")),
        "endpoint": url,
        "summary": "自行登记的评测对象。评测会保存它当次返回的回答和参考材料。",
        "command": f"python -m app.cli run --agent {name} --limit 3",
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def adapter_from_profile(profile: dict, *, top_k: int | None = None) -> AgentAdapter:
    if not profile.get("usable_with_fire_qa"):
        raise RuntimeError(f"{profile.get('title') or profile.get('name')} 不能用当前文字题库评测")
    if profile.get("adapter") == "fire_agent":
        if not os.getenv("AGENT_SHARED_TOKEN"):
            raise RuntimeError("没有读取到 AGENT_SHARED_TOKEN")
        return FireAgentAdapter(
            base_url=str(profile.get("base_url") or "http://127.0.0.1:8001"),
            top_k=top_k if top_k is not None else 5,
        )
    if profile.get("adapter") == "http_json":
        profile_top_k = profile.get("top_k")
        return HttpJsonAdapter(
            str(profile["url"]),
            query_field=str(profile.get("query_field") or "question"),
            answer_field=str(profile.get("answer_field") or "answer"),
            contexts_field=str(profile.get("contexts_field") or "contexts"),
            success_field=str(profile.get("success_field") or "status"),
            success_value=str(profile.get("success_value") or "success"),
            token_env=str(profile.get("token_env") or ""),
            token_header=str(profile.get("token_header") or ""),
            token_prefix=str(profile.get("token_prefix") or ""),
            top_k=top_k if top_k is not None else profile_top_k,
            extra_fields=dict(profile.get("extra_fields") or {}),
            name=str(profile.get("name") or "http_json"),
        )
    raise RuntimeError(f"未知适配器：{profile.get('adapter')}")


def probe_health(url: str, timeout: float = 0.4) -> str:
    if not url:
        return "未配置"
    try:
        response = httpx.get(url, timeout=timeout, trust_env=False)
    except httpx.HTTPError:
        return "未启动"
    if response.status_code == 200:
        return "在线"
    return "响应异常"


def _field(value, default: str) -> str:
    text = str(value or default).strip() or default
    if not _FIELD.fullmatch(text):
        raise ValueError("字段名只允许字母、数字、下划线和点")
    return text


def _short(value, default: str) -> str:
    text = str(value or default).strip() or default
    if len(text) > 40 or any(char in text for char in "\r\n"):
        raise ValueError("成功标记过长")
    return text


def _top_k(value) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not text.isdigit() or int(text) < 1 or int(text) > 20:
        raise ValueError("top_k 需要是 1 到 20 的整数")
    return int(text)
