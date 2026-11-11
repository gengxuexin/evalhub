"""EvalHub 命令行。

示例：
  python -m app.cli run --limit 3 --save-baseline fire_qa
  python -m app.cli show results/eval_xxx.jsonl
  python -m app.cli compare --baseline fire_qa --candidate results/eval_xxx.jsonl
  python -m app.cli rerun-failed results/eval_xxx.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.pipeline import (
    execute_evaluation,
    load_jsonl,
    save_baseline,
)
from app.adapters.registry import ADAPTER_NAMES, build_adapter
from app.catalog import adapter_from_profile, get_agent, list_agents
from app.regression import compare_runs
from app.view import publish_existing_run

EVALHUB_ROOT = Path(__file__).resolve().parents[1]
FIRE_ENV = EVALHUB_ROOT.parent / "fire_agents" / ".env"
DEFAULT_DATASET = EVALHUB_ROOT / "data" / "datasets" / "fire_qa.jsonl"
RESULTS = EVALHUB_ROOT / "results"
BASELINES = RESULTS / "baselines"
AGENTS = RESULTS / "agents"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = EVALHUB_ROOT / candidate
    return candidate


def _load_env() -> None:
    if FIRE_ENV.exists():
        load_dotenv(FIRE_ENV, override=True)


def _print_manifest(manifest: dict) -> None:
    print("run_id", manifest["run_id"])
    print("result_file", manifest["result_file"])
    print("operational_report", manifest["operational_report"])
    print("quality_report", manifest["quality_report"])
    print("failure_counts", json.dumps(manifest["failure_counts"], ensure_ascii=False))
    if manifest.get("html_report"):
        print("html_report", manifest["html_report"])


def _adapter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", default="fire_agent", choices=ADAPTER_NAMES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--url", default="", help="http_json 适配器的完整接口地址")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--contexts-field", default="contexts")
    parser.add_argument("--success-field", default="status")
    parser.add_argument("--success-value", default="success")
    parser.add_argument("--token-env", default="AGENT_SHARED_TOKEN")
    parser.add_argument("--token-header", default="X-Internal-Token")


async def _run(args) -> dict:
    if args.limit > 10 and not args.allow_large:
        raise RuntimeError("超过 10 条需要同时传入 --allow-large")

    if args.agent:
        profile = get_agent(AGENTS, args.agent)
        adapter = adapter_from_profile(profile, top_k=args.top_k)
        base_url = profile.get("base_url") or profile.get("url") or args.base_url
    else:
        adapter = build_adapter(args.adapter, args)
        base_url = args.base_url
    manifest = await execute_evaluation(
        adapter,
        _resolve(args.dataset),
        RESULTS,
        limit=args.limit,
        top_k=args.top_k,
        base_url=base_url,
    )
    _print_manifest(manifest)
    if args.save_baseline:
        path = save_baseline(args.save_baseline, manifest, BASELINES, force=args.force)
        print("baseline", path)
    return manifest


def _score(args) -> int:
    from app.metrics.providers import build_live_clients
    from app.metrics.quality import score_sample, write_quality_report
    from app.versions import suite_fingerprint

    records = load_jsonl(_resolve(args.result_file))
    if len(records) > 10 and not args.allow_large:
        raise RuntimeError("超过 10 条需要同时传入 --allow-large")
    judge = embedder = None
    if args.judge or args.semantic:
        judge, embedder = build_live_clients(str(EVALHUB_ROOT.parent / "fire_agents"))
        print("judge_model", getattr(judge, "model_name", None))
        print("embedding_model", getattr(embedder, "model_name", None))
    samples = [
        score_sample(
            record,
            judge=judge,
            embedder=embedder,
            run_judge=args.judge,
            run_semantic=args.semantic,
        )
        for record in records
    ]
    run_id = str(records[0].get("run_id") or _resolve(args.result_file).stem.removeprefix("eval_"))
    report_dir = RESULTS / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    output_name = "quality_judge_report.json" if args.judge or args.semantic else "quality_report.json"
    report_path = write_quality_report(
        samples,
        report_dir,
        source_file=str(_resolve(args.result_file)),
        run_label=run_id,
        output_name=output_name,
        versions=suite_fingerprint() if args.judge else None,
    )
    html_path = publish_existing_run(report_dir, RESULTS)
    print("quality_report", report_path)
    print("html_report", html_path)
    for sample in samples:
        print("sample", sample.get("sample_id"))
        for name, item in sample["metrics"].items():
            print(f"  {name}: status={item['status']} score={item['score']}")
    return 0


def _agree(args) -> int:
    from app.agreement import write_agreement_report
    from app.metrics.providers import build_live_clients
    from app.metrics.quality import score_sample

    if args.repeats < 2:
        raise RuntimeError("repeats 至少为 2")
    if args.repeats > 2:
        raise RuntimeError("重复评分目前最多 2 次，避免连续调用 Judge")
    records = load_jsonl(_resolve(args.result_file))
    chosen = [record for record in records if record.get("sample_id") == args.sample]
    if len(chosen) != 1:
        raise RuntimeError(f"样本 {args.sample} 需要且只能有一条")
    judge, _embedder = build_live_clients(str(EVALHUB_ROOT.parent / "fire_agents"))
    print("judge_model", judge.model_name)
    repeats = []
    for index in range(args.repeats):
        scored = score_sample(chosen[0], judge=judge, run_judge=True)
        repeats.append(scored)
        print(f"repeat {index + 1}")
        for name, item in scored["metrics"].items():
            if name in {"normalized_exact_match", "character_f1"}:
                continue
            print(f"  {name}: status={item['status']} score={item['score']}")
    run_id = str(chosen[0].get("run_id") or _resolve(args.result_file).stem.removeprefix("eval_"))
    report_path = RESULTS / "reports" / run_id / f"agreement_{args.sample}.json"
    report = write_agreement_report(report_path, sample_id=args.sample, repeats=repeats)
    print("agreement_report", report_path)
    for name, item in report["metrics"].items():
        if name in {"normalized_exact_match", "character_f1"}:
            continue
        print(f"  {name}: {item['status']}")
    return 0


def _compare(args) -> int:
    if args.baseline:
        pointer = json.loads((BASELINES / f"{args.baseline}.json").read_text(encoding="utf-8"))
        baseline_path = Path(pointer["result_file"])
    else:
        baseline_path = _resolve(args.baseline_file)
    candidate_path = _resolve(args.candidate)
    report = compare_runs(load_jsonl(baseline_path), load_jsonl(candidate_path))
    print("status", report["status"])
    for check in report["checks"]:
        print(f"  [{check['severity']}] {check['name']}: {check['message']}")
    return 1 if report["status"] == "regression" else 0


def _rerun(args) -> dict:
    records = load_jsonl(_resolve(args.result_file))
    failed = [
        record["sample_id"]
        for record in records
        if record.get("success") is not True and record.get("sample_id")
    ]
    if not failed:
        print("没有失败样本")
        return {"failed_sample_ids": []}
    print("rerun", ",".join(failed))
    adapter = build_adapter(args.adapter, args)
    manifest = asyncio.run(execute_evaluation(
        adapter,
        _resolve(args.dataset),
        RESULTS,
        sample_ids=set(failed),
        top_k=args.top_k,
        base_url=args.base_url,
    ))
    _print_manifest(manifest)
    return manifest


def _benchmark(args) -> int:
    from app.benchmarks import compare_benchmark, list_benchmarks, load_benchmark, save_benchmark

    directory = RESULTS / "benchmarks"
    if args.action == "list":
        paths = list_benchmarks(directory)
        if not paths:
            print("还没有基准")
            return 0
        for path in paths:
            print(path.stem)
        return 0
    if not args.name or not args.run:
        raise RuntimeError("save 和 check 需要 --name 与 --run")
    report_dir = RESULTS / "reports" / args.run
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到运行清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_path = report_dir / "quality_judge_report.json"
    quality = {}
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    versions = quality.get("versions") or {}
    judge_model, embedding_model = _models_from_quality(quality)
    if args.action == "save":
        path = save_benchmark(
            directory,
            args.name,
            manifest=manifest,
            versions=versions,
            judge_model=judge_model,
            embedding_model=embedding_model,
        )
        print("benchmark", path)
        return 0
    benchmark = load_benchmark(directory, args.name)
    changes = compare_benchmark(benchmark, manifest, versions)
    changed = [item for item in changes if item["status"] != "same"]
    print("status", "changed" if changed else "same")
    for item in changes:
        print(f"  [{item['status']}] {item['field']}: benchmark={item['benchmark']} run={item['run']}")
    return 1 if changed else 0


def _models_from_quality(quality: dict) -> tuple[str | None, str | None]:
    judge_model = None
    embedding_model = None
    for sample in quality.get("samples") or []:
        metrics = sample.get("metrics") or {}
        faithfulness = (metrics.get("faithfulness") or {}).get("detail") or {}
        semantic = (metrics.get("semantic_similarity") or {}).get("detail") or {}
        judge_model = judge_model or faithfulness.get("model")
        embedding_model = embedding_model or semantic.get("model")
    return judge_model, embedding_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalhub")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行小批量评测并保存结果")
    run.add_argument("--dataset", default=str(DEFAULT_DATASET))
    run.add_argument("--limit", type=int, default=3)
    run.add_argument("--agent", default="", help="使用目录中的 Agent 名称，例如 fire_agent 或 fire_gateway")
    _adapter_arguments(run)
    run.add_argument("--save-baseline")
    run.add_argument("--force", action="store_true")
    run.add_argument("--allow-large", action="store_true")

    compare = sub.add_parser("compare", help="对比基线与候选结果")
    compare.add_argument("--baseline")
    compare.add_argument("--baseline-file")
    compare.add_argument("--candidate", required=True)

    rerun = sub.add_parser("rerun-failed", help="只重跑结果文件中的失败样本")
    rerun.add_argument("result_file")
    rerun.add_argument("--dataset", default=str(DEFAULT_DATASET))
    _adapter_arguments(rerun)

    show = sub.add_parser("show", help="根据已有结果生成 HTML 报告")
    show.add_argument("source", help="结果 JSONL 或报告目录")

    score = sub.add_parser("score", help="对已有结果计算质量指标，不重新调用被测 Agent")
    score.add_argument("result_file")
    score.add_argument("--judge", action="store_true")
    score.add_argument("--semantic", action="store_true")
    score.add_argument("--allow-large", action="store_true")

    agree = sub.add_parser("agree", help="对一条已有结果重复做 Judge，检查评分是否稳定")
    agree.add_argument("result_file")
    agree.add_argument("--sample", required=True)
    agree.add_argument("--repeats", type=int, default=2)

    annotate = sub.add_parser("annotate", help="为一条样本追加人工标注")
    annotate.add_argument("run_id")
    annotate.add_argument("sample_id")
    annotate.add_argument("--label", required=True, choices=["correct", "partial", "incorrect", "skip"])
    annotate.add_argument("--comment", default="")

    sub.add_parser("agents", help="列出可接入的 Agent")

    serve = sub.add_parser("serve", help="打开本地页面，浏览结果并标注")
    serve.add_argument("--port", type=int, default=8766)

    benchmark = sub.add_parser("benchmark", help="保存或检查评测基准版本")
    benchmark.add_argument("action", choices=["save", "list", "check"])
    benchmark.add_argument("--name")
    benchmark.add_argument("--run")

    listing = sub.add_parser("baselines", help="列出已保存基线")
    listing.set_defaults(command="baselines")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        asyncio.run(_run(args))
        return 0
    if args.command == "compare":
        if bool(args.baseline) == bool(args.baseline_file):
            raise RuntimeError("compare 需要且只能指定 --baseline 或 --baseline-file 其中一个")
        return _compare(args)
    if args.command == "rerun-failed":
        _rerun(args)
        return 0
    if args.command == "show":
        path = publish_existing_run(_resolve(args.source), RESULTS)
        print("html_report", path)
        return 0
    if args.command == "score":
        return _score(args)
    if args.command == "agree":
        return _agree(args)
    if args.command == "annotate":
        from app.annotations import append_annotation

        record = append_annotation(
            RESULTS / "annotations",
            run_id=args.run_id,
            sample_id=args.sample_id,
            label=args.label,
            comment=args.comment,
        )
        print("annotated", record["sample_id"], record["label"])
        return 0
    if args.command == "serve":
        import uvicorn

        from app.web import create_app

        uvicorn.run(create_app(RESULTS), host="127.0.0.1", port=args.port)
        return 0
    if args.command == "agents":
        for item in list_agents(AGENTS):
            mark = "可评测" if item.get("usable_with_fire_qa") else "当前题库不适用"
            print(item["name"], mark, item.get("endpoint") or item.get("url") or "")
        return 0
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "baselines":
        if not BASELINES.exists():
            print("还没有基线")
            return 0
        for path in sorted(BASELINES.glob("*.json")):
            print(path.name)
        return 0
    raise RuntimeError(f"未知命令：{args.command}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
