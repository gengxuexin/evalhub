"""答案与参考答案的余弦相似度。

分数就是余弦值，范围理论上为 [-1, 1]。
这里不设通过阈值，也不把它换算成正确概率。
"""

import math
from typing import Any, Protocol

from app.metrics.contracts import (
    METRIC_FAILURE,
    NOT_EVALUABLE,
    SCORED,
    metric_result,
    MetricResult,
)


SEMANTIC_DEFINITION = (
    "使用指定向量模型计算 answer 与 ground_truth 的余弦相似度。"
    "未设置通过阈值，余弦值不是校准后的正确概率。"
)


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序返回同样数量的向量。"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("向量维度不一致或为空")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("零向量无法计算余弦相似度")
    return dot / (left_norm * right_norm)


def semantic_similarity(
    record: dict[str, Any],
    embedder: Embedder | None,
) -> MetricResult:
    if embedder is None:
        return metric_result(
            "semantic_similarity",
            NOT_EVALUABLE,
            detail={
                "reason": "embedder_not_configured",
                "definition": SEMANTIC_DEFINITION,
                "pass_threshold": None,
            },
            error="未配置向量模型",
        )

    if record.get("success") is not True:
        return metric_result(
            "semantic_similarity",
            NOT_EVALUABLE,
            detail={
                "reason": "agent_call_failed",
                "definition": SEMANTIC_DEFINITION,
                "model": getattr(embedder, "model_name", None),
                "pass_threshold": None,
            },
            error="被测 Agent 调用未成功",
        )

    answer = record.get("answer")
    reference = record.get("ground_truth")
    if (
        not isinstance(answer, str)
        or not isinstance(reference, str)
        or not answer.strip()
        or not reference.strip()
    ):
        return metric_result(
            "semantic_similarity",
            NOT_EVALUABLE,
            detail={
                "reason": "missing_text",
                "definition": SEMANTIC_DEFINITION,
                "model": embedder.model_name,
                "pass_threshold": None,
            },
            error="answer 或 ground_truth 为空",
        )

    try:
        vectors = embedder.embed([answer, reference])
        if len(vectors) != 2:
            raise ValueError(f"向量数量应为 2，实际 {len(vectors)}")
        score = cosine_similarity(vectors[0], vectors[1])
    except Exception as exc:
        return metric_result(
            "semantic_similarity",
            METRIC_FAILURE,
            detail={
                "definition": SEMANTIC_DEFINITION,
                "model": embedder.model_name,
                "pass_threshold": None,
            },
            error=f"{type(exc).__name__}: {exc}",
        )

    return metric_result(
        "semantic_similarity",
        SCORED,
        score=score,
        detail={
            "definition": SEMANTIC_DEFINITION,
            "model": embedder.model_name,
            "score_range": "[-1, 1]",
            "pass_threshold": None,
            "rescaled_to_probability": False,
        },
    )
