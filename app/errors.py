"""把失败样本分成可复跑的错误类别。成功样本记为 ok。"""


def classify_failure(record: dict) -> str:
    if record.get("success") is True:
        return "ok"

    error = str(record.get("error") or "")
    lowered = error.lower()
    status = record.get("status_code")

    if status == 401 or "401" in error or "token" in lowered or "认证" in error:
        return "auth_failed"
    if "timeout" in lowered or "超时" in error:
        return "timeout"
    if any(word in lowered for word in ("connect", "connection", "requesterror")) or "请求异常" in error:
        return "connection_error"
    if isinstance(status, int) and status >= 500:
        return "http_server_error"
    if isinstance(status, int) and status >= 400:
        return "http_client_error"
    if status == 200 and not str(record.get("answer") or "").strip():
        return "empty_answer"
    if error:
        return "agent_error"
    return "unknown_failure"


def count_failures(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = record.get("error_class") or classify_failure(record)
        counts[label] = counts.get(label, 0) + 1
    return counts


def failed_sample_ids(records: list[dict]) -> list[str]:
    return [
        record["sample_id"]
        for record in records
        if record.get("success") is not True and record.get("sample_id")
    ]
