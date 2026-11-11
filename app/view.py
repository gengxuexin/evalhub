"""把一次评测渲染成单个 HTML 文件，方便直接打开查看。"""

import html
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: Any, digits: int = 1) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{digits}f}"


def render_html(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    operational: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
) -> str:
    operational = operational or {}
    quality = quality or {}
    metrics = operational.get("metrics") or {}
    summary = quality.get("summary") or {}
    quality_by_id = {
        item.get("sample_id"): item.get("metrics") or {}
        for item in quality.get("samples") or []
    }
    failures = [item for item in records if item.get("success") is not True]

    cards = [
        ("样本数", metrics.get("total", len(records))),
        ("接口成功", metrics.get("success_count", "-")),
        ("接口失败", metrics.get("failure_count", "-")),
        ("平均耗时 ms", _num(metrics.get("average_latency_ms"))),
        ("P95 耗时 ms", _num(metrics.get("p95_latency_ms"))),
    ]
    card_html = "".join(
        f"<div class='card'><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>"
        for label, value in cards
    )

    quality_rows = []
    for name, item in summary.items():
        quality_rows.append(
            "<tr>"
            f"<td>{_esc(name)}</td>"
            f"<td>{_esc(item.get('scored_count'))}</td>"
            f"<td>{_num(item.get('mean_score'), 3)}</td>"
            f"<td>{_esc(item.get('status_counts'))}</td>"
            "</tr>"
        )

    sample_rows = []
    for record in records:
        sample_metrics = quality_by_id.get(record.get("sample_id"), {})
        exact = (sample_metrics.get("normalized_exact_match") or {}).get("score")
        f1 = (sample_metrics.get("character_f1") or {}).get("score")
        contexts = record.get("contexts") or []
        context_count = len(contexts) if isinstance(contexts, list) else 0
        answer = str(record.get("answer") or "")
        context_html = "".join(f"<pre>{_esc(text)}</pre>" for text in contexts if isinstance(text, str))
        sample_rows.append(
            "<tr>"
            f"<td>{_esc(record.get('sample_id'))}</td>"
            f"<td>{'成功' if record.get('success') is True else '失败'}</td>"
            f"<td>{_esc(record.get('error_class') or '')}</td>"
            f"<td>{_num(record.get('latency_ms'))}</td>"
            f"<td>{context_count}</td>"
            f"<td>{_num(exact, 3)}</td>"
            f"<td>{_num(f1, 3)}</td>"
            f"<td>{_esc(record.get('question'))}</td>"
            "</tr>"
            "<tr class='detail'><td colspan='8'>"
            f"<p><b>参考答案</b> {_esc(record.get('ground_truth'))}</p>"
            f"<p><b>回答</b></p><pre>{_esc(answer)}</pre>"
            f"<p><b>当次上下文</b> {context_count} 条</p>{context_html or '<p>无</p>'}"
            f"<p><b>错误</b> {_esc(record.get('error') or '无')}</p>"
            "</td></tr>"
        )

    failure_rows = "".join(
        "<li>"
        f"{_esc(item.get('sample_id'))} / {_esc(item.get('error_class') or 'unknown')} / {_esc(item.get('error'))}"
        "</li>"
        for item in failures
    ) or "<li>没有失败样本</li>"
    warnings = "".join(f"<li>{_esc(item)}</li>" for item in quality.get("warnings") or [])

    dataset = manifest.get("dataset") or {}
    adapter = manifest.get("adapter") or {}
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>EvalHub {_esc(manifest.get("run_id"))}</title>
<style>
html, body {{ background: #ffffff; color: #1f2933; color-scheme: light; }}
body {{ font-family: "Segoe UI", sans-serif; margin: 24px; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.muted {{ color: #52606d; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ background: #f5f7fa; border-radius: 8px; padding: 12px 16px; min-width: 120px; }}
.card span {{ display: block; color: #52606d; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border-bottom: 1px solid #e4e7eb; text-align: left; padding: 8px; vertical-align: top; }}
th {{ background: #f5f7fa; }}
pre {{ white-space: pre-wrap; background: #f8fafc; padding: 8px; }}
.detail td {{ background: #fcfdfe; }}
</style>
</head>
<body>
<h1>EvalHub 评测结果 {_esc(manifest.get("run_id"))}</h1>
<p class="muted">接口成功率不是答案正确率。字面分只表示文本重合。</p>
<p>数据集 {_esc(dataset.get("path"))}<br>
样本 { _esc(dataset.get("executed_samples")) } / {_esc(dataset.get("total_samples")) }，
SHA256 {_esc(dataset.get("sha256"))}<br>
适配器 {_esc(adapter.get("name"))}，地址 {_esc(adapter.get("base_url"))}，top_k {_esc(adapter.get("top_k"))}</p>
<div class="cards">{card_html}</div>
<h2>质量指标</h2>
<table>
<tr><th>指标</th><th>已评分数</th><th>平均分</th><th>状态计数</th></tr>
{''.join(quality_rows) or '<tr><td colspan="4">没有质量汇总</td></tr>'}
</table>
<h2>说明</h2>
<ul>{warnings or '<li>无</li>'}</ul>
<h2>失败样本</h2>
<ul>{failure_rows}</ul>
<h2>逐条结果</h2>
<table>
<tr><th>样本</th><th>调用</th><th>错误类别</th><th>耗时 ms</th><th>上下文</th><th>精确匹配</th><th>字符 F1</th><th>问题</th></tr>
{''.join(sample_rows)}
</table>
</body>
</html>
"""


def write_html_report(
    path: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    operational: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html(manifest, records, operational, quality),
        encoding="utf-8",
    )
    return path


def publish_existing_run(source: Path, results_dir: Path) -> Path:
    """根据已有 JSONL 或报告目录生成 HTML，不重新调用 Agent。"""

    if source.is_dir():
        manifest = _load(source / "manifest.json")
        result_file = Path(manifest["result_file"])
        report_dir = source
    else:
        records = _load_jsonl(source)
        run_id = records[0].get("run_id") or source.stem.removeprefix("eval_")
        report_dir = results_dir / "reports" / str(run_id)
        manifest = _load(report_dir / "manifest.json")
        result_file = source
        if not manifest:
            manifest = {
                "run_id": run_id,
                "result_file": str(source),
                "dataset": {},
                "adapter": {},
                "failure_counts": {},
                "failed_sample_ids": [],
            }
    records = _load_jsonl(result_file)
    operational = _load(manifest.get("operational_report") or report_dir / "operational_report.json")
    quality = _merge_quality(
        _load(manifest.get("quality_report") or report_dir / "quality_report.json"),
        _load(report_dir / "quality_judge_report.json"),
    )
    return write_html_report(report_dir / "report.html", manifest, records, operational, quality)


def _merge_quality(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    if not extra:
        return base
    if not base:
        return extra
    samples = {
        item.get("sample_id"): {"sample_id": item.get("sample_id"), "metrics": dict(item.get("metrics") or {})}
        for item in base.get("samples") or []
    }
    for item in extra.get("samples") or []:
        target = samples.setdefault(
            item.get("sample_id"),
            {"sample_id": item.get("sample_id"), "metrics": {}},
        )
        target["metrics"].update(item.get("metrics") or {})
    warnings = list(base.get("warnings") or [])
    for warning in extra.get("warnings") or []:
        if warning not in warnings:
            warnings.append(warning)
    summary = dict(base.get("summary") or {})
    summary.update(extra.get("summary") or {})
    merged = dict(base)
    merged["samples"] = list(samples.values())
    merged["summary"] = summary
    merged["warnings"] = warnings
    if extra.get("versions"):
        merged["versions"] = extra["versions"]
    return merged


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    from app.pipeline import load_jsonl

    return load_jsonl(path)
