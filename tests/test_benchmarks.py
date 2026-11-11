import json
import tempfile
import unittest
from pathlib import Path

from app.agreement import repeat_agreement
from app.benchmarks import compare_benchmark, save_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_records_config_and_detects_topk_change(self):
        manifest = {
            "run_id": "run_1",
            "result_file": "results/eval_run_1.jsonl",
            "dataset": {"sha256": "aaa", "path": "data/fire_qa.jsonl", "executed_samples": 3},
            "adapter": {"name": "fire_agent", "base_url": "http://127.0.0.1:8001", "top_k": 5},
        }
        versions = {"suite_version": "2026-11-11.2", "judge_prompt_sha256": "prompt"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = save_benchmark(
                directory,
                "fire_qa_v1",
                manifest=manifest,
                versions=versions,
                judge_model="qwen-plus",
                embedding_model="text-embedding-v4",
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset_sha256"], "aaa")
            self.assertEqual(saved["judge_model"], "qwen-plus")
            with self.assertRaises(FileExistsError):
                save_benchmark(directory, "fire_qa_v1", manifest=manifest)
            changed = dict(manifest)
            changed["adapter"] = dict(manifest["adapter"])
            changed["adapter"]["top_k"] = 8
            diffs = compare_benchmark(saved, changed, versions)
            topk = [item for item in diffs if item["field"] == "top_k"][0]
            self.assertEqual(topk["status"], "changed")
            dataset = [item for item in diffs if item["field"] == "dataset_sha256"][0]
            self.assertEqual(dataset["status"], "same")

    def test_repeat_agreement_marks_label_conflict(self):
        first = {"metrics": {"answer_relevancy": {"status": "scored", "score": 1.0, "detail": {"label": "direct"}}}}
        second = {"metrics": {"answer_relevancy": {"status": "scored", "score": 0.5, "detail": {"label": "partial"}}}}
        report = repeat_agreement([first, second])
        self.assertEqual(report["metrics"]["answer_relevancy"]["status"], "inconsistent")

    def test_agreement_report_renders_consistent_metric(self):
        from app.agreement import write_agreement_report
        from app.web import _agreement_table

        repeats = [
            {"sample_id": "sample_0001", "metrics": {"faithfulness": {"status": "scored", "score": 1.0, "detail": {}}}},
            {"sample_id": "sample_0001", "metrics": {"faithfulness": {"status": "scored", "score": 1.0, "detail": {}}}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = write_agreement_report(
                directory / "agreement_sample_0001.json",
                sample_id="sample_0001",
                repeats=repeats,
            )
            self.assertEqual(report["metrics"]["faithfulness"]["status"], "consistent")
            page = _agreement_table(directory, "run_1")
            self.assertIn("是否依据材料", page)
            self.assertIn("一致", page)
        first = {"metrics": {"faithfulness": {"status": "scored", "score": 1.0, "detail": {}}}}
        second = {"metrics": {"faithfulness": {"status": "judge_failure", "score": None, "detail": {}}}}
        report = repeat_agreement([first, second])
        self.assertEqual(report["metrics"]["faithfulness"]["status"], "not_comparable")


if __name__ == "__main__":
    unittest.main()
