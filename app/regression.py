"""对比两次评测。接口成功率和字面质量分开判断，缺失指标不记成退步。"""

from app.metrics.basic_metrics import BasicMetrics
from app.metrics.quality import score_sample, summarize_quality


DEFAULT_RULES = {
    "success_rate_drop": 0.0,
    "latency_increase_ratio": 0.5,
    "quality_drop": 0.1,
}


def compare_runs(
    baseline: list[dict],
    candidate: list[dict],
    rules: dict | None = None,
) -> dict:
    active = {**DEFAULT_RULES, **(rules or {})}
    checks = []
    checks.extend(_operational_checks(baseline, candidate, active))
    checks.extend(_quality_checks(baseline, candidate, active))
    checks.extend(_sample_checks(baseline, candidate))

    if any(item["severity"] == "regression" for item in checks):
        status = "regression"
    elif any(item["severity"] == "warning" for item in checks):
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "rules": active,
        "checks": checks,
        "new_failures": [
            item for item in checks if item["name"] == "new_failure"
        ],
        "recovered": [
            item for item in checks if item["name"] == "recovered"
        ],
    }


def _operational_checks(baseline: list[dict], candidate: list[dict], rules: dict) -> list[dict]:
    if not baseline or not candidate:
        return [_check("run_present", "warning", "基线或候选结果为空，无法对比")]

    base_metrics = BasicMetrics.calculate(baseline)
    new_metrics = BasicMetrics.calculate(candidate)
    checks = []

    drop = base_metrics["success_rate"] - new_metrics["success_rate"]
    if drop > rules["success_rate_drop"]:
        checks.append(_check(
            "success_rate",
            "regression",
            f"成功率从 {base_metrics['success_rate']:.2%} 降到 {new_metrics['success_rate']:.2%}",
            baseline=base_metrics["success_rate"],
            candidate=new_metrics["success_rate"],
        ))
    else:
        checks.append(_check(
            "success_rate",
            "ok",
            "接口成功率没有下降",
            baseline=base_metrics["success_rate"],
            candidate=new_metrics["success_rate"],
        ))

    base_latency = base_metrics["average_latency_ms"]
    new_latency = new_metrics["average_latency_ms"]
    if base_latency in (None, 0) or new_latency is None:
        checks.append(_check("average_latency_ms", "warning", "耗时不足，不能判断是否变慢"))
    else:
        ratio = (new_latency - base_latency) / base_latency
        severity = "warning" if ratio > rules["latency_increase_ratio"] else "ok"
        checks.append(_check(
            "average_latency_ms",
            severity,
            f"平均耗时变化 {ratio:.0%}",
            baseline=base_latency,
            candidate=new_latency,
        ))
    return checks


def _quality_checks(baseline: list[dict], candidate: list[dict], rules: dict) -> list[dict]:
    base_summary = summarize_quality([score_sample(item) for item in baseline])
    new_summary = summarize_quality([score_sample(item) for item in candidate])
    checks = []
    for name in sorted(set(base_summary) | set(new_summary)):
        base_item = base_summary.get(name) or {}
        new_item = new_summary.get(name) or {}
        base_score = base_item.get("mean_score")
        new_score = new_item.get("mean_score")
        if base_score is None or new_score is None:
            checks.append(_check(
                name,
                "warning",
                "有一侧没有可平均的得分，不把缺失当成退步",
                baseline=base_score,
                candidate=new_score,
            ))
            continue
        drop = base_score - new_score
        severity = "warning" if drop > rules["quality_drop"] else "ok"
        message = f"平均分变化 {-drop:+.3f}"
        if severity == "warning":
            message += "。字面质量波动只作提示，不单独判定回归"
        checks.append(_check(
            name,
            severity,
            message,
            baseline=base_score,
            candidate=new_score,
        ))
    return checks


def _sample_checks(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    base_map = {item.get("sample_id"): item for item in baseline}
    new_map = {item.get("sample_id"): item for item in candidate}
    checks = []
    overlap = [sample_id for sample_id in base_map if sample_id in new_map]
    if not overlap:
        checks.append(_check("sample_overlap", "warning", "没有可对齐的 sample_id"))
        return checks

    for sample_id in overlap:
        before = base_map[sample_id].get("success") is True
        after = new_map[sample_id].get("success") is True
        if before and not after:
            checks.append(_check(
                "new_failure",
                "regression",
                f"{sample_id} 由成功变为失败",
                sample_id=sample_id,
                error=new_map[sample_id].get("error"),
            ))
        elif after and not before:
            checks.append(_check(
                "recovered",
                "ok",
                f"{sample_id} 由失败变为成功",
                sample_id=sample_id,
            ))
    return checks


def _check(name: str, severity: str, message: str, **extra) -> dict:
    return {"name": name, "severity": severity, "message": message, **extra}
