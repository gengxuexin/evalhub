
"""
EvalHub 统一 Agent 接口定义。

所有被测 Agent 都需要遵循这个接口，
从而让评测执行引擎不依赖具体 Agent 的实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    """统一的 Agent 请求数据结构。"""

    query: str
    sample_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """统一的 Agent 响应数据结构。"""

    answer: str
    success: bool

    # 执行信息
    latency_ms: float = 0.0
    status_code: int | None = None
    error: str | None = None

    # 模型与评测信息
    model: str | None = None
    contexts: list[str] = field(default_factory=list)

    # 保留原始响应，便于后续深入分析
    raw_response: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(ABC):
    """
    Agent 适配器抽象基类。

    不同领域的 Agent 通过继承该类，
    实现自己的 invoke() 方法。
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def invoke(self, request: AgentRequest) -> AgentResponse:
        """
        调用被测 Agent。

        Args:
            request: 统一格式的请求对象

        Returns:
            统一格式的响应对象
        """
        raise NotImplementedError
