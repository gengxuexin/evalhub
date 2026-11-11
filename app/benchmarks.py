"""评测基准版本。

记录一次运行所使用的数据集、适配器、top_k 和 Judge 提示词版本。
后续运行可以用这些字段检查配置是否还能对照。
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _safe_name(name: str) -> str:
    safe = "".join(char for char in name if char.isalnum() or char in ("_", "-"))
    if not safe or safe != name:
        raise ValueError("基准名称含有不允许的字符")
    return safe


def save_benchmark(
    directory: Path,
    name: str,
    *,
    manifest: dict,
    versions: dict | None = None,
    judge_model: str | None = None,
    embedding_model: str | None = None,
    force: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(name)}.json"
    if path.exists() and not force:
        raise FileExistsError(f"基准已存在：{path}")
    dataset = manifest.get("dataset") or {}
    adapter = manifest.get("adapter") or {}
    versions = versions or {}
    payload = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": manifest.get("run_id"),
        "result_file": manifest.get("result_file"),
        "dataset_sha256": dataset.get("sha256"),
        "dataset_path": dataset.get("path"),
        "executed_samples": dataset.get("executed_samples"),
        "adapter_name": adapter.get("name"),
        "base_url": adapter.get("base_url"),
        "top_k": adapter.get("top_k"),
        "suite_version": versions.get("suite_version"),
        "judge_prompt_sha256": versions.get("judge_prompt_sha256"),
        "judge_model": judge_model,
        "embedding_model": embedding_model,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_benchmark(directory: Path, name: str) -> dict:
    path = directory / f"{_safe_name(name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"基准不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_benchmarks(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def compare_benchmark(benchmark: dict, manifest: dict, versions: dict | None = None) -> list[dict]:
    """列出和基准不一致的配置。不一致不等于结果退步。"""

    dataset = manifest.get("dataset") or {}
    adapter = manifest.get("adapter") or {}
    versions = versions or {}
    pairs = [
        ("dataset_sha256", benchmark.get("dataset_sha256"), dataset.get("sha256")),
        ("adapter_name", benchmark.get("adapter_name"), adapter.get("name")),
        ("top_k", benchmark.get("top_k"), adapter.get("top_k")),
        ("suite_version", benchmark.get("suite_version"), versions.get("suite_version")),
        ("judge_prompt_sha256", benchmark.get("judge_prompt_sha256"), versions.get("judge_prompt_sha256")),
    ]
    changes = []
    for field, expected, actual in pairs:
        if expected is None or actual is None:
            changes.append({
                "field": field,
                "status": "missing",
                "benchmark": expected,
                "run": actual,
            })
            continue
        changes.append({
            "field": field,
            "status": "same" if expected == actual else "changed",
            "benchmark": expected,
            "run": actual,
        })
    return changes
