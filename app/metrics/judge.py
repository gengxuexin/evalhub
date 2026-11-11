"""基于当次上下文的 Judge 指标。

只使用结果记录里已经保存的 contexts。
本模块不会重新检索，也不会在失败时返回 0 分。
"""

import json
import re
from typing import Any, Protocol

from app.metrics.contracts import (
    INVALID_OUTPUT,
    JUDGE_ABSTAIN,
    JUDGE_FAILURE,
    NOT_EVALUABLE,
    SCORED,
    metric_result,
    MetricResult,
)


MAX_CONTEXT_CHARS = 1500
MAX_ANSWER_CHARS = 3000

FAITHFULNESS_SYSTEM = """你是严谨的评测器。只根据给定的检索上下文判断回答中的原子命题是否被支持。
不得使用上下文之外的知识补全。
主题相关、同一部法律的其他条款、或更宽泛的背景，都不算支持。
只有上下文明确表达了同一命题时才算 supported；同义改写可以，推断新事实不行。
证据不足时 verdict 必须是 insufficient，不能记成 unsupported，也不能算作支持。"""

RELEVANCY_SYSTEM = """你判断回答是否切题。只评估回答是否在回应问题，不评估事实是否正确。
只能返回三档之一：direct、partial、irrelevant。
不要输出 0 到 1 的自由分数，不要生成多个问题再计算命中率。
无法判断时返回 abstain。"""

RELEVANCE_RATIO_SYSTEM = """你逐条判断检索片段是否含有回答该问题所需的信息。
relevant 只表示这一条片段本身是否相关，不考虑它的排序位置，也不要根据命中名次加权。
证据不足时不要猜成相关。"""

RECALL_SYSTEM = """你判断检索上下文是否覆盖参考答案里的必要事实。
先把参考答案拆成不能再拆的原子事实。
covered 只在上下文明确包含同一命题时使用，允许同义改写。
主题相关不等于覆盖。同一部法律的其他条款、更宽泛的表述、或只能间接推断的内容，都是 not_covered。
无法从上下文判断时使用 insufficient，不得把它算成 covered。"""


class JudgeClient(Protocol):
    model_name: str

    def complete(self, system: str, user: str) -> str:
        """返回模型原文。JSON 解析由指标函数负责。"""


def parse_judge_payload(raw: str) -> dict[str, Any]:
    """解析 Judge 输出。失败时抛出 ValueError，由调用方记为 invalid_output。"""

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge 输出不是 JSON 对象") from None
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge 输出必须是 JSON 对象")
    return payload


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _agent_ready(record: dict[str, Any], name: str) -> MetricResult | None:
    if record.get("success") is not True:
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "agent_call_failed"},
            error="被测 Agent 调用未成功",
        )
    answer = record.get("answer")
    question = record.get("question")
    if not isinstance(answer, str) or not answer.strip():
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "empty_answer"},
            error="回答为空",
        )
    if not isinstance(question, str) or not question.strip():
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "missing_question"},
            error="问题为空",
        )
    return None


def _context_block(record: dict[str, Any], name: str) -> tuple[list[str], dict] | MetricResult:
    """只读取记录中的 contexts，不发起第二次检索。"""

    raw = record.get("raw_response")
    contexts = record.get("contexts")
    if contexts is None and isinstance(raw, dict) and "contexts" not in raw:
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "contexts_not_returned"},
            error="当次响应没有 contexts 字段，不能评检索质量",
        )
    if not isinstance(contexts, list):
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "invalid_contexts"},
            error="contexts 不是列表",
        )
    if not contexts:
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "empty_contexts"},
            error="contexts 为空。缺失上下文不记 0 分，也不记满分",
        )
    cleaned = []
    truncated = []
    for index, item in enumerate(contexts):
        if not isinstance(item, str) or not item.strip():
            return metric_result(
                name,
                NOT_EVALUABLE,
                detail={"reason": "invalid_contexts"},
                error=f"第 {index} 条 context 不是非空字符串",
            )
        text, was_truncated = _clip(item.strip(), MAX_CONTEXT_CHARS)
        cleaned.append(text)
        if was_truncated:
            truncated.append(index)
    return cleaned, {"truncated_context_indexes": truncated}


def _call_judge(
    name: str,
    judge: JudgeClient | None,
    system: str,
    user: str,
) -> dict[str, Any] | MetricResult:
    if judge is None:
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "judge_not_configured"},
            error="未配置 Judge",
        )
    try:
        raw = judge.complete(system, user)
    except Exception as exc:
        return metric_result(
            name,
            JUDGE_FAILURE,
            detail={"model": getattr(judge, "model_name", None)},
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        payload = parse_judge_payload(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return metric_result(
            name,
            INVALID_OUTPUT,
            detail={
                "model": judge.model_name,
                "raw_excerpt": raw[:300],
            },
            error=str(exc),
        )
    if payload.get("decision") == "abstain":
        return metric_result(
            name,
            JUDGE_ABSTAIN,
            detail={
                "model": judge.model_name,
                "reason": payload.get("reason"),
            },
            error="Judge 主动拒评",
        )
    return payload


def faithfulness(
    record: dict[str, Any],
    judge: JudgeClient | None,
) -> MetricResult:
    name = "faithfulness"
    blocked = _agent_ready(record, name)
    if blocked:
        return blocked
    contexts = _context_block(record, name)
    if isinstance(contexts, MetricResult):
        return contexts
    context_texts, truncation = contexts

    numbered = "\n\n".join(
        f"[{index}] {text}" for index, text in enumerate(context_texts, start=1)
    )
    answer, answer_truncated = _clip(record["answer"].strip(), MAX_ANSWER_CHARS)
    user = (
        f"问题：{record['question'].strip()}\n"
        f"回答：{answer}\n"
        f"检索上下文：\n{numbered}\n"
        "只输出 JSON："
        '{"claims":["原子命题"],'
        '"verdicts":[{"claim_index":0,'
        '"verdict":"supported|unsupported|insufficient",'
        '"reason":"依据哪一条上下文，或为何证据不足"}]}'
        "。verdicts 数量必须等于 claims，claim_index 从 0 连续。"
    )
    payload = _call_judge(name, judge, FAITHFULNESS_SYSTEM, user)
    if isinstance(payload, MetricResult):
        return payload

    claims = payload.get("claims")
    verdicts = payload.get("verdicts")
    if not isinstance(claims, list) or not claims or not all(
        isinstance(item, str) and item.strip() for item in claims
    ):
        return metric_result(
            name,
            INVALID_OUTPUT,
            detail={"model": judge.model_name, "reason": "claims_missing"},
            error="claims 必须是非空字符串列表",
        )
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        return metric_result(
            name,
            INVALID_OUTPUT,
            detail={
                "model": judge.model_name,
                "reason": "verdict_count_mismatch",
                "claim_count": len(claims),
                "verdict_count": len(verdicts) if isinstance(verdicts, list) else None,
            },
            error="verdict 数量与命题数量不一致",
        )

    allowed = {"supported", "unsupported", "insufficient"}
    counts = {key: 0 for key in allowed}
    for index, item in enumerate(verdicts):
        if not isinstance(item, dict) or item.get("claim_index") != index:
            return metric_result(
                name,
                INVALID_OUTPUT,
                detail={"model": judge.model_name, "reason": "claim_index_invalid"},
                error=f"第 {index} 个 verdict 的 claim_index 不正确",
            )
        verdict = item.get("verdict")
        if verdict not in allowed:
            return metric_result(
                name,
                INVALID_OUTPUT,
                detail={"model": judge.model_name, "reason": "verdict_label_invalid"},
                error=f"非法 verdict：{verdict}",
            )
        counts[verdict] += 1

    decided = counts["supported"] + counts["unsupported"]
    detail = {
        "definition": (
            "回答中的原子命题有多少被当次 contexts 支持。"
            "insufficient 不进入分母，也不记为 0。"
        ),
        "model": judge.model_name,
        "claim_count": len(claims),
        "supported": counts["supported"],
        "unsupported": counts["unsupported"],
        "insufficient": counts["insufficient"],
        "used_second_retrieval": False,
        "answer_truncated": answer_truncated,
        **truncation,
    }
    if decided == 0:
        return metric_result(
            name,
            JUDGE_ABSTAIN,
            detail=detail,
            error="全部命题都是证据不足，没有可计算的支持率",
        )
    return metric_result(
        name,
        SCORED,
        score=counts["supported"] / decided,
        detail=detail,
    )


_RELEVANCY_SCORES = {"direct": 1.0, "partial": 0.5, "irrelevant": 0.0}


def answer_relevancy(
    record: dict[str, Any],
    judge: JudgeClient | None,
) -> MetricResult:
    name = "answer_relevancy"
    blocked = _agent_ready(record, name)
    if blocked:
        return blocked

    answer, answer_truncated = _clip(record["answer"].strip(), MAX_ANSWER_CHARS)
    user = (
        f"问题：{record['question'].strip()}\n"
        f"回答：{answer}\n"
        "只输出 JSON："
        '{"label":"direct|partial|irrelevant","reason":"..."}'
        "。无法判断时输出 "
        '{"decision":"abstain","reason":"..."}。'
    )
    payload = _call_judge(name, judge, RELEVANCY_SYSTEM, user)
    if isinstance(payload, MetricResult):
        return payload

    label = payload.get("label")
    if label not in _RELEVANCY_SCORES:
        return metric_result(
            name,
            INVALID_OUTPUT,
            detail={"model": judge.model_name, "reason": "label_invalid"},
            error=f"非法切题档位：{label}",
        )
    return metric_result(
        name,
        SCORED,
        score=_RELEVANCY_SCORES[label],
        detail={
            "definition": "回答是否切题。三档序数映射为 direct=1、partial=0.5、irrelevant=0。",
            "label": label,
            "reason": payload.get("reason"),
            "model": judge.model_name,
            "scale": "3-level ordinal mapped to {0, 0.5, 1}",
            "divided_by_generated_question_count": False,
            "is_calibrated_probability": False,
            "answer_truncated": answer_truncated,
        },
    )


def _average_precision(flags: list[bool]) -> float:
    """按原顺序计算平均精度。没有相关片段时返回 0。"""

    relevant = 0
    total = 0.0
    for rank, flag in enumerate(flags, start=1):
        if flag:
            relevant += 1
            total += relevant / rank
    if relevant == 0:
        return 0.0
    return total / relevant


def _with_precision_failure(result: MetricResult) -> dict[str, MetricResult]:
    result.name = "context_relevance_ratio"
    return {
        "context_relevance_ratio": result,
        "context_average_precision": metric_result(
            "context_average_precision",
            result.status,
            detail=dict(result.detail),
            error=result.error,
        ),
    }


def retrieval_context_metrics(
    record: dict[str, Any],
    judge: JudgeClient | None,
) -> dict[str, MetricResult]:
    """一次 Judge 调用同时给出相关比例和按原顺序的平均精度。"""

    name = "context_relevance_ratio"
    blocked = _agent_ready(record, name)
    if blocked:
        return _with_precision_failure(blocked)
    contexts = _context_block(record, name)
    if isinstance(contexts, MetricResult):
        return _with_precision_failure(contexts)
    context_texts, truncation = contexts

    numbered = "\n\n".join(
        f"[{index}] {text}" for index, text in enumerate(context_texts)
    )
    user = (
        f"问题：{record['question'].strip()}\n"
        f"检索上下文：\n{numbered}\n"
        "只输出 JSON："
        '{"verdicts":[{"context_index":0,"relevant":true,"reason":"..."}]}'
        f"。verdicts 数量必须等于 {len(context_texts)}，"
        "context_index 从 0 连续。不要按排序加权。"
    )
    payload = _call_judge(name, judge, RELEVANCE_RATIO_SYSTEM, user)
    if isinstance(payload, MetricResult):
        return _with_precision_failure(payload)

    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(context_texts):
        return _with_precision_failure(metric_result(
            name,
            INVALID_OUTPUT,
            detail={
                "model": judge.model_name,
                "reason": "verdict_count_mismatch",
                "context_count": len(context_texts),
                "verdict_count": len(verdicts) if isinstance(verdicts, list) else None,
            },
            error="相关片段判定数量与上下文数量不一致",
        ))

    flags = []
    relevant = 0
    for index, item in enumerate(verdicts):
        if not isinstance(item, dict) or item.get("context_index") != index:
            return _with_precision_failure(metric_result(
                name,
                INVALID_OUTPUT,
                detail={"model": judge.model_name, "reason": "context_index_invalid"},
                error=f"第 {index} 个 verdict 的 context_index 不正确",
            ))
        if not isinstance(item.get("relevant"), bool):
            return _with_precision_failure(metric_result(
                name,
                INVALID_OUTPUT,
                detail={"model": judge.model_name, "reason": "relevant_not_bool"},
                error=f"第 {index} 个 relevant 不是布尔值",
            ))
        flags.append(item["relevant"])
        relevant += int(item["relevant"])

    shared = {
        "relevant_count": relevant,
        "context_count": len(context_texts),
        "model": judge.model_name,
        **truncation,
    }
    return {
        "context_relevance_ratio": metric_result(
            "context_relevance_ratio",
            SCORED,
            score=relevant / len(context_texts),
            detail={
                "definition": "相关片段占全部检索片段的比例。不考虑排序。",
                "metric_is_rank_aware_context_precision": False,
                "considers_rank": False,
                **shared,
            },
        ),
        "context_average_precision": metric_result(
            "context_average_precision",
            SCORED,
            score=_average_precision(flags),
            detail={
                "definition": (
                    "按上下文原顺序计算的平均精度。"
                    "每个相关片段贡献其所在位置的 precision，再除以相关片段数。"
                    "没有相关片段时得分为 0。"
                ),
                "considers_rank": True,
                "rank_source": "原检索顺序，没有重新排序",
                **shared,
            },
        ),
    }


def context_relevance_ratio(
    record: dict[str, Any],
    judge: JudgeClient | None,
) -> MetricResult:
    """相关片段比例。不考虑排序，因此不命名为 context precision。"""

    return retrieval_context_metrics(record, judge)["context_relevance_ratio"]


def context_recall(
    record: dict[str, Any],
    judge: JudgeClient | None,
) -> MetricResult:
    name = "context_recall"
    blocked = _agent_ready(record, name)
    if blocked:
        return blocked
    reference = record.get("ground_truth")
    if not isinstance(reference, str) or not reference.strip():
        return metric_result(
            name,
            NOT_EVALUABLE,
            detail={"reason": "empty_reference"},
            error="参考答案为空",
        )
    contexts = _context_block(record, name)
    if isinstance(contexts, MetricResult):
        return contexts
    context_texts, truncation = contexts

    numbered = "\n\n".join(
        f"[{index}] {text}" for index, text in enumerate(context_texts, start=1)
    )
    reference_text, reference_truncated = _clip(reference.strip(), MAX_ANSWER_CHARS)
    user = (
        f"问题：{record['question'].strip()}\n"
        f"参考答案：{reference_text}\n"
        f"检索上下文：\n{numbered}\n"
        "主题相关不等于覆盖。"
        "只输出 JSON："
        '{"facts":["原子事实"],'
        '"verdicts":[{"fact_index":0,'
        '"verdict":"covered|not_covered|insufficient",'
        '"reason":"..."}]}'
        "。verdicts 数量必须等于 facts，fact_index 从 0 连续。"
    )
    payload = _call_judge(name, judge, RECALL_SYSTEM, user)
    if isinstance(payload, MetricResult):
        return payload

    facts = payload.get("facts")
    verdicts = payload.get("verdicts")
    if not isinstance(facts, list) or not facts or not all(
        isinstance(item, str) and item.strip() for item in facts
    ):
        return metric_result(
            name,
            INVALID_OUTPUT,
            detail={"model": judge.model_name, "reason": "facts_missing"},
            error="facts 必须是非空字符串列表",
        )
    if not isinstance(verdicts, list) or len(verdicts) != len(facts):
        return metric_result(
            name,
            INVALID_OUTPUT,
            detail={
                "model": judge.model_name,
                "reason": "verdict_count_mismatch",
                "fact_count": len(facts),
                "verdict_count": len(verdicts) if isinstance(verdicts, list) else None,
            },
            error="verdict 数量与事实数量不一致",
        )

    allowed = {"covered", "not_covered", "insufficient"}
    counts = {key: 0 for key in allowed}
    for index, item in enumerate(verdicts):
        if not isinstance(item, dict) or item.get("fact_index") != index:
            return metric_result(
                name,
                INVALID_OUTPUT,
                detail={"model": judge.model_name, "reason": "fact_index_invalid"},
                error=f"第 {index} 个 verdict 的 fact_index 不正确",
            )
        verdict = item.get("verdict")
        if verdict not in allowed:
            return metric_result(
                name,
                INVALID_OUTPUT,
                detail={"model": judge.model_name, "reason": "verdict_label_invalid"},
                error=f"非法 verdict：{verdict}",
            )
        counts[verdict] += 1

    decided = counts["covered"] + counts["not_covered"]
    detail = {
        "definition": (
            "参考答案的原子事实有多少被当次 contexts 明确覆盖。"
            "主题相关不算覆盖。insufficient 不进入分母。"
        ),
        "model": judge.model_name,
        "fact_count": len(facts),
        "covered": counts["covered"],
        "not_covered": counts["not_covered"],
        "insufficient": counts["insufficient"],
        "topic_overlap_counts_as_coverage": False,
        "reference_truncated": reference_truncated,
        **truncation,
    }
    if decided == 0:
        return metric_result(
            name,
            JUDGE_ABSTAIN,
            detail=detail,
            error="全部事实都无法判断，没有可计算的覆盖率",
        )
    return metric_result(
        name,
        SCORED,
        score=counts["covered"] / decided,
        detail=detail,
    )
