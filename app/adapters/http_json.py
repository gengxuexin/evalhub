"""通用 HTTP JSON 适配器。

字段名由调用方指定，因此可以接入消防 Agent 以外的服务。
Token 只从指定的环境变量读取，不会写入结果。
"""

import os
import time
from typing import Any

import httpx

from app.adapters.base import AgentAdapter, AgentRequest, AgentResponse


class HttpJsonAdapter(AgentAdapter):
    def __init__(
        self,
        url: str,
        *,
        query_field: str = "question",
        answer_field: str = "answer",
        contexts_field: str = "contexts",
        success_field: str = "status",
        success_value: str = "success",
        token_env: str = "",
        token_header: str = "X-Internal-Token",
        token_prefix: str = "",
        top_k: int | None = None,
        extra_fields: dict[str, Any] | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        name: str = "http_json",
    ):
        super().__init__(name=name)
        self.url = url
        self.query_field = query_field
        self.answer_field = answer_field
        self.contexts_field = contexts_field
        self.success_field = success_field
        self.success_value = success_value
        self.token_env = token_env
        self.token_header = token_header
        self.token_prefix = token_prefix
        self.top_k = top_k
        self.extra_fields = dict(extra_fields or {})
        self.timeout = timeout
        self._client = client

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        payload: dict[str, Any] = {self.query_field: request.query}
        payload.update(self.extra_fields)
        if self.top_k is not None:
            payload["top_k"] = self.top_k

        headers = {"Content-Type": "application/json"}
        if self.token_env:
            token = os.getenv(self.token_env, "")
            if token and self.token_header:
                headers[self.token_header] = f"{self.token_prefix}{token}"

        start = time.perf_counter()
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        owns_client = self._client is None
        try:
            response = await client.post(self.url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                return AgentResponse(
                    answer="",
                    success=False,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    error="响应不是 JSON 对象",
                    raw_response={"body": body},
                )
            answer = _lookup(body, self.answer_field)
            if not isinstance(answer, str):
                answer = "" if answer is None else str(answer)
            contexts = _lookup(body, self.contexts_field)
            if not isinstance(contexts, list):
                contexts = []
            success_mark = _lookup(body, self.success_field)
            success = success_mark == self.success_value and bool(answer.strip())
            return AgentResponse(
                answer=answer,
                success=success,
                latency_ms=latency_ms,
                status_code=response.status_code,
                error=None if success else str(body.get("message") or "Agent 返回失败或答案为空"),
                model=body.get("model") if isinstance(body.get("model"), str) else None,
                contexts=[item for item in contexts if isinstance(item, str)],
                raw_response=body,
            )
        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return AgentResponse(
                answer="",
                success=False,
                latency_ms=latency_ms,
                status_code=exc.response.status_code,
                error=f"HTTP 错误: {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return AgentResponse(
                answer="",
                success=False,
                latency_ms=latency_ms,
                error=f"请求异常: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return AgentResponse(
                answer="",
                success=False,
                latency_ms=latency_ms,
                error=f"未知异常: {type(exc).__name__}: {exc}",
            )
        finally:
            if owns_client:
                await client.aclose()


def _lookup(body: dict[str, Any], path: str) -> Any:
    current: Any = body
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
