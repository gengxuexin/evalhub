"""字面重合指标。

normalized exact match 和 character F1 只描述规范化后的字面关系。
它们不是语义正确率，也不能代替接口成功率。
"""

import unicodedata
from collections import Counter
from typing import Any

from app.metrics.contracts import (
    NOT_EVALUABLE,
    SCORED,
    metric_result,
    MetricResult,
)


EXACT_MATCH_DEFINITION = (
    "NFKC 规范化并去掉标点和空白后的字符串全等。"
    "回答比参考答案更长但包含同一事实时仍为 0。"
    "这不是语义正确率。"
)
CHARACTER_F1_DEFINITION = (
    "规范化后按字符多重集计算的 F1。"
    "中文没有分词，每个剩余字符算一个 token。"
    "这只表示字面重合，不等于语义正确。"
)


def normalize_answer(text: str) -> str:
    """去掉标点、空白，并做 NFKC 与 ASCII 小写。"""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    kept = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("Z"):
            continue
        kept.append(char)
    return "".join(kept)


def _prepare_pair(record: dict[str, Any]) -> tuple[str, str] | MetricResult:
    if record.get("success") is not True:
        return metric_result(
            "",
            NOT_EVALUABLE,
            detail={"reason": "agent_call_failed"},
            error="被测 Agent 调用未成功，不能把失败记成字面 0 分",
        )

    answer = record.get("answer")
    reference = record.get("ground_truth")
    if not isinstance(answer, str) or not isinstance(reference, str):
        return metric_result(
            "",
            NOT_EVALUABLE,
            detail={"reason": "missing_text"},
            error="answer 或 ground_truth 不是字符串",
        )
    return answer, reference


def normalized_exact_match(record: dict[str, Any]) -> MetricResult:
    prepared = _prepare_pair(record)
    if isinstance(prepared, MetricResult):
        prepared.name = "normalized_exact_match"
        prepared.detail = {
            **prepared.detail,
            "definition": EXACT_MATCH_DEFINITION,
        }
        return prepared

    answer, reference = prepared
    left = normalize_answer(answer)
    right = normalize_answer(reference)
    if not right:
        return metric_result(
            "normalized_exact_match",
            NOT_EVALUABLE,
            detail={
                "reason": "empty_reference",
                "definition": EXACT_MATCH_DEFINITION,
            },
            error="参考答案规范化后为空",
        )

    return metric_result(
        "normalized_exact_match",
        SCORED,
        score=1.0 if left == right else 0.0,
        detail={
            "definition": EXACT_MATCH_DEFINITION,
            "suitable_as_primary_quality_score": False,
            "normalized_answer_length": len(left),
            "normalized_reference_length": len(right),
        },
    )


def character_f1(record: dict[str, Any]) -> MetricResult:
    prepared = _prepare_pair(record)
    if isinstance(prepared, MetricResult):
        prepared.name = "character_f1"
        prepared.detail = {
            **prepared.detail,
            "definition": CHARACTER_F1_DEFINITION,
        }
        return prepared

    answer, reference = prepared
    left = normalize_answer(answer)
    right = normalize_answer(reference)
    if not right:
        return metric_result(
            "character_f1",
            NOT_EVALUABLE,
            detail={
                "reason": "empty_reference",
                "definition": CHARACTER_F1_DEFINITION,
            },
            error="参考答案规范化后为空",
        )

    predicted = Counter(left)
    expected = Counter(right)
    overlap = sum((predicted & expected).values())
    if not predicted and not expected:
        precision = recall = f1 = 1.0
    else:
        precision = overlap / sum(predicted.values()) if predicted else 0.0
        recall = overlap / sum(expected.values()) if expected else 0.0
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

    return metric_result(
        "character_f1",
        SCORED,
        score=f1,
        detail={
            "definition": CHARACTER_F1_DEFINITION,
            "precision": precision,
            "recall": recall,
            "score_range": "[0, 1]",
            "equals_semantic_correctness": False,
        },
    )
