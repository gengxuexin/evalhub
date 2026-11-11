import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.annotations import append_annotation, latest_annotations
from app.web import create_app


class AnnotationTests(unittest.TestCase):
    def test_latest_label_wins_and_unknown_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            append_annotation(directory, run_id="run_1", sample_id="sample_0001", label="incorrect", comment="第一次")
            append_annotation(directory, run_id="run_1", sample_id="sample_0001", label="correct", comment="改判")
            latest = latest_annotations(directory, "run_1")
            self.assertEqual(latest["sample_0001"]["label"], "correct")
            self.assertEqual(latest["sample_0001"]["comment"], "改判")
            with self.assertRaises(ValueError):
                append_annotation(directory, run_id="run_1", sample_id="sample_0001", label="yes")

    def test_web_saves_annotation_for_known_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "run_demo"
            report_dir = root / "reports" / run_id
            report_dir.mkdir(parents=True)
            result = root / "eval_run_demo.jsonl"
            result.write_text(json.dumps({
                "run_id": run_id,
                "sample_id": "sample_0001",
                "success": True,
                "question": "问题",
                "answer": "回答",
                "ground_truth": "参考",
                "contexts": ["片段"],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest = {
                "run_id": run_id,
                "result_file": str(result),
                "dataset": {"executed_samples": 1},
                "failure_counts": {"ok": 1},
            }
            (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (report_dir / "quality_judge_report.json").write_text(json.dumps({
                "summary": {"faithfulness": {"scored_count": 1, "mean_score": 1.0}},
                "versions": {"suite_version": "test-suite", "judge_prompt_sha256": "abc"},
                "samples": [{
                    "sample_id": "sample_0001",
                    "metrics": {"faithfulness": {"status": "scored", "score": 1.0}},
                }],
            }), encoding="utf-8")
            client = TestClient(create_app(root))
            page = client.get(f"/runs/{run_id}")
            self.assertEqual(page.status_code, 200)
            self.assertIn("sample_0001", page.text)
            self.assertIn("是否依据材料", page.text)
            self.assertIn("评分规则", page.text)
            saved = client.post("/annotations", data={
                "run_id": run_id,
                "sample_id": "sample_0001",
                "label": "partial",
                "comment": "缺了一句",
            }, follow_redirects=False)
            self.assertEqual(saved.status_code, 303)
            again = client.get(f"/runs/{run_id}")
            self.assertIn("partial", again.text)
            stored = latest_annotations(root / "annotations", run_id)
            self.assertEqual(stored["sample_0001"]["comment"], "缺了一句")

    def test_site_pages_list_fire_adapter_without_running_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(Path(tmp)))
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn("消防评测", home.text)
            self.assertIn("评测题库", home.text)
            agents = client.get("/agents")
            self.assertEqual(agents.status_code, 200)
            self.assertIn("法规问答", agents.text)
            self.assertIn("登记其他评测对象", agents.text)
            standards = client.get("/standards")
            self.assertEqual(standards.status_code, 200)
            self.assertIn("是否依据材料", standards.text)
            benchmarks = client.get("/benchmarks")
            self.assertEqual(benchmarks.status_code, 200)
            styles = client.get("/static/vendor/css/app.css")
            self.assertEqual(styles.status_code, 200)


if __name__ == "__main__":
    unittest.main()
