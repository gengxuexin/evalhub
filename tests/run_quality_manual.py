"""在 WSL 中计算质量指标。

默认先用历史结果做无模型的字面指标。
传入 --live 时额外做 1 条真实 Agent 调用，再对这一条做 Judge 和向量相似度。
不会覆盖已有结果，也不会跑完整数据集。
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path("/home/geng/fire_agent_v2")
EVALHUB = ROOT / "EvalHub"
FIRE_AGENTS = ROOT / "fire_agents"
DATASET = EVALHUB / "data" / "datasets" / "fire_qa.jsonl"
HISTORY = EVALHUB / "results" / "eval_20261110_125549_68f4e5f9.jsonl"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _print_summary(title: str, report_path: Path, samples: list[dict]) -> None:
    print(f"\n===== {title} =====")
    print("report", report_path)
    print("samples", len(samples))
    for sample in samples:
        print("sample", sample.get("sample_id"))
        for name, item in sample["metrics"].items():
            print(
                f"  {name}: status={item['status']} score={item['score']} "
                f"reason={(item.get('detail') or {}).get('reason')}"
            )


def score_file(path: Path, output_dir: Path, run_label: str, **kwargs) -> Path:
    from app.metrics.quality import score_sample, write_quality_report

    rows = _load_jsonl(path)
    samples = [score_sample(row, **kwargs) for row in rows]
    report_path = write_quality_report(
        samples,
        output_dir,
        source_file=str(path),
        run_label=run_label,
    )
    _print_summary(run_label, report_path, samples)
    return report_path


async def run_one_sample() -> Path:
    from app.adapters.fire_agent import FireAgentAdapter
    from app.datasets.loader import DatasetLoader
    from app.runner.executor import EvaluationExecutor
    from app.storage.result_store import ResultStore

    if not os.getenv("AGENT_SHARED_TOKEN"):
        raise RuntimeError("没有读取到 AGENT_SHARED_TOKEN")

    samples = DatasetLoader().load_jsonl(str(DATASET))
    adapter = FireAgentAdapter(base_url="http://127.0.0.1:8001", top_k=5)
    results = await EvaluationExecutor(adapter).run_batch(samples, limit=1)
    output_path = ResultStore(output_dir=str(EVALHUB / "results")).save_results(results)
    item = results[0]
    print(
        "live_agent",
        item["sample_id"],
        "success",
        item["success"],
        "contexts",
        len(item.get("contexts") or []),
        "answer_len",
        len(item.get("answer") or ""),
        "file",
        output_path.name,
    )
    if item.get("success") is not True or not item.get("contexts"):
        raise RuntimeError(item.get("error") or "真实调用没有返回 contexts")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(EVALHUB))
    os.chdir(EVALHUB)
    load_dotenv(FIRE_AGENTS / ".env", override=True)

    stamp = _stamp()
    score_file(
        HISTORY,
        EVALHUB / "results" / "reports" / f"quality_lexical_{stamp}",
        "historical_lexical_only",
    )

    if not args.live:
        print("LEXICAL_ONLY_DONE")
        return

    live_path = asyncio.run(run_one_sample())
    from app.metrics.providers import build_live_clients

    judge, embedder = build_live_clients(str(FIRE_AGENTS))
    print("judge_model", judge.model_name)
    print("embedding_model", embedder.model_name)
    score_file(
        live_path,
        EVALHUB / "results" / "reports" / f"quality_live_{stamp}",
        "live_one_sample",
        judge=judge,
        embedder=embedder,
        run_judge=True,
        run_semantic=True,
    )
    print("LIVE_QUALITY_DONE")


if __name__ == "__main__":
    main()
