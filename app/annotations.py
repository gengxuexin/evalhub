"""人工标注。同一条样本可以多次标注，展示时使用最后一次。"""

import json
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("correct", "partial", "incorrect", "skip")


def annotation_path(directory: Path, run_id: str) -> Path:
    safe = "".join(char for char in run_id if char.isalnum() or char in ("_", "-"))
    if not safe or safe != run_id:
        raise ValueError("run_id 含有不允许的字符")
    return directory / f"{safe}.jsonl"


def append_annotation(
    directory: Path,
    *,
    run_id: str,
    sample_id: str,
    label: str,
    comment: str = "",
    annotator: str = "local",
) -> dict:
    if label not in LABELS:
        raise ValueError(f"标注标签必须是 {', '.join(LABELS)}")
    if not sample_id:
        raise ValueError("sample_id 不能为空")
    record = {
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "sample_id": sample_id,
        "label": label,
        "comment": comment.strip(),
        "annotator": annotator.strip() or "local",
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = annotation_path(directory, run_id)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def latest_annotations(directory: Path, run_id: str) -> dict[str, dict]:
    path = annotation_path(directory, run_id)
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict) and item.get("sample_id"):
            latest[item["sample_id"]] = item
    return latest
