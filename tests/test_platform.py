import json
import tempfile
import unittest
from pathlib import Path

from app.adapters.base import AgentResponse
from app.errors import classify_failure, failed_sample_ids
from app.pipeline import execute_evaluation, load_jsonl, save_baseline, select_samples
from app.datasets.loader import DatasetLoader
from app.regression import compare_runs


class FakeAdapter:
    name = "fake"

    def __init__(self, fail_ids=None):
        self.fail_ids = set(fail_ids or [])

    async def invoke(self, request):
        if request.sample_id in self.fail_ids:
            return AgentResponse(
                answer="",
                success=False,
                latency_ms=12,
                status_code=500,
                error="HTTP 错误: 500",
            )
        return AgentResponse(
            answer=request.query,
            success=True,
            latency_ms=20,
            status_code=200,
            contexts=["ctx"],
            raw_response={"status": "success", "contexts": ["ctx"]},
        )


def _write_dataset(directory: Path) -> Path:
    path = directory / "qa.jsonl"
    rows = [
        {"question": "问题一", "ground_truth": "问题一", "category": "c", "difficulty": "easy", "verified": True},
        {"question": "问题二", "ground_truth": "问题二", "category": "c", "difficulty": "easy", "verified": False},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


class PlatformTests(unittest.TestCase):
    def test_error_classes(self):
        self.assertEqual(classify_failure({"success": True}), "ok")
        self.assertEqual(classify_failure({"success": False, "status_code": 401, "error": "no"}), "auth_failed")
        self.assertEqual(classify_failure({"success": False, "error": "Timeout"}), "timeout")
        self.assertEqual(classify_failure({"success": False, "status_code": 500, "error": "HTTP 错误: 500"}), "http_server_error")
        self.assertEqual(
            failed_sample_ids([
                {"sample_id": "sample_0001", "success": True},
                {"sample_id": "sample_0002", "success": False},
            ]),
            ["sample_0002"],
        )

    def test_select_failed_ids_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_dataset(Path(tmp))
            samples = DatasetLoader().load_jsonl(str(dataset))
            chosen = select_samples(samples, sample_ids={"sample_0002"})
            self.assertEqual([sample.sample_id for sample in chosen], ["sample_0002"])

    def test_pipeline_saves_distinct_files_and_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _write_dataset(root)
            manifest = __import__("asyncio").run(execute_evaluation(
                FakeAdapter(fail_ids={"sample_0002"}),
                dataset,
                root / "results",
                limit=2,
                top_k=5,
                base_url="http://fake",
            ))
            records = load_jsonl(Path(manifest["result_file"]))
            self.assertEqual([item["error_class"] for item in records], ["ok", "http_server_error"])
            self.assertTrue(Path(manifest["operational_report"]).exists())
            self.assertTrue(Path(manifest["quality_report"]).exists())
            baseline = save_baseline("demo", manifest, root / "baselines")
            with self.assertRaises(FileExistsError):
                save_baseline("demo", manifest, root / "baselines")
            self.assertTrue(baseline.exists())

    def test_success_drop_is_regression_and_quality_drop_is_warning(self):
        baseline = [
            {"sample_id": "sample_0001", "success": True, "latency_ms": 100, "answer": "消防法", "ground_truth": "消防法"},
            {"sample_id": "sample_0002", "success": True, "latency_ms": 100, "answer": "告知承诺", "ground_truth": "告知承诺"},
        ]
        candidate = [
            {"sample_id": "sample_0001", "success": False, "latency_ms": 100, "answer": "", "ground_truth": "消防法", "error": "down"},
            {"sample_id": "sample_0002", "success": True, "latency_ms": 100, "answer": "完全不同的长回答", "ground_truth": "告知承诺"},
        ]
        report = compare_runs(baseline, candidate)
        self.assertEqual(report["status"], "regression")
        names = {item["name"] for item in report["checks"] if item["severity"] == "regression"}
        self.assertIn("success_rate", names)
        self.assertIn("new_failure", names)
        quality = [item for item in report["checks"] if item["name"] == "character_f1"][0]
        self.assertEqual(quality["severity"], "warning")

    def test_identical_runs_pass(self):
        records = [
            {"sample_id": "sample_0001", "success": True, "latency_ms": 100, "answer": "消防法", "ground_truth": "消防法"},
        ]
        self.assertEqual(compare_runs(records, records)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
