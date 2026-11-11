"""质量指标的统一返回契约。

未得分的指标 score 必须为 None。汇总时只对 status=scored 取平均，
因此评测器失败、拒评和证据缺失不会被当成 0 分。
"""

import math
from dataclasses import dataclass, field
from typing import Any


SCORED = "scored"
NOT_APPLICABLE = "not_applicable"
NOT_EVALUABLE = "not_evaluable"
JUDGE_FAILURE = "judge_failure"
JUDGE_ABSTAIN = "judge_abstain"
INVALID_OUTPUT = "invalid_output"
METRIC_FAILURE = "metric_failure"

UNSCORED_STATUSES = {
    NOT_APPLICABLE,
    NOT_EVALUABLE,
    JUDGE_FAILURE,
    JUDGE_ABSTAIN,
    INVALID_OUTPUT,
    METRIC_FAILURE,
}


@dataclass
class MetricResult:
    """单条样本上一个指标的结果。"""

    name: str
    status: str
    score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "score": self.score,
            "detail": self.detail,
            "error": self.error,
        }


def metric_result(
    name: str,
    status: str,
    *,
    score: float | None = None,
    detail: dict[str, Any] | None = None,
    error: str | None = None,
) -> MetricResult:
    """构造并校验指标结果，避免未得分状态携带可被误平均的分数。"""

    if status == SCORED:
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"{name} 得分状态必须带有限数值 score")
        score = float(score)
    elif status in UNSCORED_STATUSES:
        if score is not None:
            raise ValueError(f"{name} 在状态 {status} 下不能携带 score")
        score = None
    else:
        raise ValueError(f"未知指标状态：{status}")

    return MetricResult(
        name=name,
        status=status,
        score=score,
        detail=detail or {},
        error=error,
    )
