"""把单条评测记录上的质量指标汇总成可追溯报告。"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.metrics.contracts import SCORED, MetricResult
from app.metrics.judge import (
    answer_relevancy,
    context_recall,
    faithfulness,
    retrieval_context_metrics,
)
from app.metrics.lexical import character_f1, normalized_exact_match
from app.metrics.semantic import semantic_similarity


WARNINGS = [
    "接口成功率不是答案正确率。",
    "normalized_exact_match 和 character_f1 只表示字面重合。",
    "semantic_similarity 是指定模型的余弦值，没有通过阈值。",
    "answer_relevancy 是三档序数映射，不是把命中数除以 5。",
    "context_relevance_ratio 是相关片段比例，不考虑排序。",
    "context_average_precision 按原检索顺序计算，没有相关片段时为 0。",
    "contexts 缺失或为空时，依赖上下文的指标为 not_evaluable，不是 0 分或满分。",
    "Judge 失败、输出不合法和主动拒评都不进入平均分。",
]


def score_sample(
    record: dict[str, Any],
    *,
    judge=None,
    embedder=None,
    run_judge: bool = False,
    run_semantic: bool = False,
) -> dict[str, Any]:
    """为一条样本计算质量指标。单条异常不会抛出到批次外。"""

    metrics: dict[str, MetricResult] = {
        "normalized_exact_match": normalized_exact_match(record),
        "character_f1": character_f1(record),
    }

    if run_semantic:
        metrics["semantic_similarity"] = semantic_similarity(record, embedder)
    if run_judge:
        metrics["faithfulness"] = faithfulness(record, judge)
        metrics["answer_relevancy"] = answer_relevancy(record, judge)
        metrics.update(retrieval_context_metrics(record, judge))
        metrics["context_recall"] = context_recall(record, judge)

    return {
        "sample_id": record.get("sample_id"),
        "metrics": {
            name: result.to_dict() for name, result in metrics.items()
        },
    }


def summarize_quality(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """只对 scored 样本取平均。其他状态单独计数。"""

    names: list[str] = []
    for sample in samples:
        for name in sample.get("metrics", {}):
            if name not in names:
                names.append(name)

    summary = {}
    for name in names:
        statuses: Counter[str] = Counter()
        scores: list[float] = []
        labels: Counter[str] = Counter()
        for sample in samples:
            item = sample.get("metrics", {}).get(name)
            if not isinstance(item, dict):
                continue
            status = item.get("status") or "missing"
            statuses[status] += 1
            if status == SCORED and isinstance(item.get("score"), (int, float)):
                scores.append(float(item["score"]))
            label = (item.get("detail") or {}).get("label")
            if isinstance(label, str):
                labels[label] += 1

        summary[name] = {
            "scored_count": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "status_counts": dict(statuses),
            "label_counts": dict(labels),
        }
    return summary


def write_quality_report(
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    source_file: str,
    run_label: str,
    output_name: str = "quality_report.json",
    versions: dict[str, Any] | None = None,
) -> Path:
    """写入质量报告。默认文件名保持不变，Judge 结果使用单独文件名。"""

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "source_file": source_file,
        "sample_count": len(samples),
        "warnings": WARNINGS,
        "summary": summarize_quality(samples),
        "samples": samples,
    }
    if versions:
        report["versions"] = versions
    output_path = report_dir / output_name
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
