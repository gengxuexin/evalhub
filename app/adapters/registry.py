"""按名称创建被测 Agent 适配器。执行器只依赖统一接口。"""

import os

from app.adapters.base import AgentAdapter
from app.adapters.fire_agent import FireAgentAdapter
from app.adapters.http_json import HttpJsonAdapter


ADAPTER_NAMES = ("fire_agent", "http_json")


def build_adapter(name: str, options) -> AgentAdapter:
    if name == "fire_agent":
        if not os.getenv("AGENT_SHARED_TOKEN"):
            raise RuntimeError("没有读取到 AGENT_SHARED_TOKEN")
        return FireAgentAdapter(base_url=options.base_url, top_k=options.top_k)

    if name == "http_json":
        if not options.url:
            raise RuntimeError("http_json 适配器需要 --url")
        return HttpJsonAdapter(
            url=options.url,
            query_field=options.query_field,
            answer_field=options.answer_field,
            contexts_field=options.contexts_field,
            success_field=options.success_field,
            success_value=options.success_value,
            token_env=options.token_env,
            token_header=options.token_header,
            top_k=options.top_k,
        )

    raise RuntimeError(f"未知适配器：{name}")
