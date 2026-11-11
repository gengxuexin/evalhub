"""同一样本多次 Judge 的一致性。只比较已有评分，不发起新调用。"""


def repeat_agreement(repeats: list[dict]) -> dict:
    """repeats 是同一样本的多次 score_sample 结果。"""

    if len(repeats) < 2:
        raise ValueError("一致性至少需要两次评分")
    names: list[str] = []
    for repeat in repeats:
        for name in repeat.get("metrics") or {}:
            if name not in names:
                names.append(name)

    metrics = {}
    for name in names:
        items = [(repeat.get("metrics") or {}).get(name) or {} for repeat in repeats]
        statuses = [item.get("status") for item in items]
        scores = [item.get("score") for item in items]
        labels = [(item.get("detail") or {}).get("label") for item in items]
        if any(status != "scored" for status in statuses) or any(score is None for score in scores):
            metrics[name] = {
                "status": "not_comparable",
                "reason": "存在未得分的重复评测",
                "statuses": statuses,
            }
            continue
        numeric = [float(score) for score in scores]
        label_values = [label for label in labels if label is not None]
        labels_match = len(label_values) <= 1 or len(set(label_values)) == 1
        scores_match = max(numeric) - min(numeric) <= 1e-9
        metrics[name] = {
            "status": "consistent" if scores_match and labels_match else "inconsistent",
            "min_score": min(numeric),
            "max_score": max(numeric),
            "labels": label_values,
        }
    return {"repeat_count": len(repeats), "metrics": metrics}


def write_agreement_report(
    path,
    *,
    sample_id: str,
    repeats: list[dict],
) -> dict:
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "note": "同一条已保存的 Agent 回答被重复评分，没有重新调用被测 Agent。",
        **repeat_agreement(repeats),
        "repeats": repeats,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
