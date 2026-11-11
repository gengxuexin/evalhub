
"""
消防 Agent 适配器。

负责将 EvalHub 的统一请求转换为消防 Agent 的 HTTP 请求，
并将消防 Agent 的响应转换为统一的 AgentResponse。
"""
import os
import time
from typing import Any

import httpx

from app.adapters.base import AgentAdapter, AgentRequest, AgentResponse


class FireAgentAdapter(AgentAdapter):
    """消防法规 RAG Agent 适配器。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        internal_token: str | None = None,
        timeout: float = 120.0,
        top_k: int = 5,
    ):
        super().__init__(name="fire_agent")

        self.base_url = base_url.rstrip("/")
        self.internal_token = (
    internal_token
    if internal_token is not None
    else os.getenv("AGENT_SHARED_TOKEN", "")
)
        self.timeout = timeout
        self.top_k = top_k

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        """调用消防 Agent 的 RAG 问答接口。"""

        url = f"{self.base_url}/api/rag/ask"

        headers = {
            "Content-Type": "application/json"
        }

        # 如果消防 Agent 配置了内部调用 Token，就携带认证信息
        if self.internal_token:
            headers["X-Internal-Token"] = self.internal_token

        payload = {
            "question": request.query,
            "top_k": self.top_k,
        }

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                )

                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                response.raise_for_status()

                result: dict[str, Any] = response.json()

                answer = result.get("answer", "")
                success = (
                    result.get("status") == "success"
                    and bool(answer)
                )

                return AgentResponse(
                    answer=answer,
                    success=success,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    error=None if success else result.get(
                        "message", "Agent 返回失败或答案为空"
                    ),
                    model=result.get("model"),
                    contexts=result.get("contexts", []),
                    raw_response=result,
                )

        except httpx.HTTPStatusError as exc:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return AgentResponse(
                answer="",
                success=False,
                latency_ms=latency_ms,
                status_code=exc.response.status_code,
                error=f"HTTP 错误: {exc.response.status_code}",
            )

        except httpx.RequestError as exc:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return AgentResponse(
                answer="",
                success=False,
                latency_ms=latency_ms,
                error=f"请求异常: {type(exc).__name__}: {exc}",
            )

        except Exception as exc:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return AgentResponse(
                answer="",
                success=False,
                latency_ms=latency_ms,
                error=f"未知异常: {type(exc).__name__}: {exc}",
            )
