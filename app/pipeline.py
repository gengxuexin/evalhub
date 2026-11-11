"""一次可复现的评测运行：执行、落盘、运行指标、字面质量、基线。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from app.datasets.loader import DatasetLoader, DatasetSample
from app.errors import classify_failure, count_failures, failed_sample_ids
from app.metrics.quality import score_sample, write_quality_report
from app.metrics.report import ReportGenerator
from app.runner.executor import EvaluationExecutor
from app.storage.result_store import ResultStore
from app.view import write_html_report


def dataset_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def select_samples(
    samples: Sequence[DatasetSample],
    *,
    limit: int | None = None,
    sample_ids: set[str] | None = None,
) -> list[DatasetSample]:
    chosen = list(samples)
    if sample_ids is not None:
        chosen = [sample for sample in chosen if sample.sample_id in sample_ids]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        chosen = chosen[:limit]
    return chosen


async def execute_evaluation(
    adapter,
    dataset_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    sample_ids: set[str] | None = None,
    top_k: int | None = None,
    base_url: str | None = None,
) -> dict:
    """执行评测并写入独立结果目录。不覆盖旧结果。"""

    samples = DatasetLoader().load_jsonl(str(dataset_path))
    chosen = select_samples(samples, limit=limit, sample_ids=sample_ids)
    if not chosen:
        raise ValueError("没有可执行的样本")

    results = await EvaluationExecutor(adapter).run_batch(chosen)
    for record in results:
        record["error_class"] = classify_failure(record)

    result_path = ResultStore(output_dir=str(output_dir)).save_results(results)
    run_id = result_path.stem.removeprefix("eval_")
    report_dir = output_dir / "reports" / run_id

    operational = ReportGenerator().generate(
        str(result_path),
        output_dir=str(report_dir),
        output_name="operational_report.json",
    )
    quality_samples = [score_sample(record) for record in results]
    quality = write_quality_report(
        quality_samples,
        report_dir,
        source_file=str(result_path),
        run_label=run_id,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            **dataset_fingerprint(dataset_path),
            "total_samples": len(samples),
            "executed_samples": len(chosen),
        },
        "adapter": {
            "name": getattr(adapter, "name", None),
            "base_url": base_url or getattr(adapter, "url", None),
            "top_k": top_k,
        },
        "result_file": str(result_path),
        "operational_report": str(operational),
        "quality_report": str(quality),
        "failure_counts": count_failures(results),
        "failed_sample_ids": failed_sample_ids(results),
    }
    html_report = write_html_report(
        report_dir / "report.html",
        manifest,
        results,
        json.loads(operational.read_text(encoding="utf-8")),
        json.loads(quality.read_text(encoding="utf-8")),
    )
    manifest["html_report"] = str(html_report)
    manifest_path = report_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_file"] = str(manifest_path)
    return manifest


def save_baseline(name: str, manifest: dict, baseline_dir: Path, force: bool = False) -> Path:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / f"{name}.json"
    if path.exists() and not force:
        raise FileExistsError(f"基线已存在：{path}")
    payload = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "result_file": manifest["result_file"],
        "manifest_file": manifest["manifest_file"],
        "dataset": manifest["dataset"],
        "adapter": manifest["adapter"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
        rows.append(item)
    if not rows:
        raise ValueError(f"结果文件为空：{path}")
    return rows
