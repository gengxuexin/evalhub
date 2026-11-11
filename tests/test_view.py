import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx

from app.adapters.http_json import HttpJsonAdapter
from app.adapters.registry import build_adapter
from app.view import render_html


class Options:
    def __init__(self, **kwargs):
        self.base_url = "http://127.0.0.1:8001"
        self.url = ""
        self.top_k = 5
        self.query_field = "question"
        self.answer_field = "answer"
        self.contexts_field = "contexts"
        self.success_field = "status"
        self.success_value = "success"
        self.token_env = "AGENT_SHARED_TOKEN"
        self.token_header = "X-Internal-Token"
        for key, value in kwargs.items():
            setattr(self, key, value)


class ViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_adapter_maps_fields_and_keeps_token_out_of_response(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode())
            seen["header_present"] = "X-Test-Token" in request.headers
            return httpx.Response(200, json={
                "status": "ok",
                "output": "回答<script>",
                "docs": ["片段一"],
            })

        os.environ["EVALHUB_TEST_TOKEN"] = "secret-token"
        self.addCleanup(os.environ.pop, "EVALHUB_TEST_TOKEN", None)
        adapter = HttpJsonAdapter(
            "http://example.test/ask",
            query_field="q",
            answer_field="output",
            contexts_field="docs",
            success_field="status",
            success_value="ok",
            token_env="EVALHUB_TEST_TOKEN",
            token_header="X-Test-Token",
            top_k=2,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        from app.adapters.base import AgentRequest

        response = await adapter.invoke(AgentRequest(query="问题", sample_id="s1"))
        self.assertTrue(response.success)
        self.assertEqual(response.contexts, ["片段一"])
        self.assertEqual(seen["body"]["q"], "问题")
        self.assertTrue(seen["header_present"])
        self.assertNotIn("secret-token", json.dumps(response.raw_response))

    def test_registry_requires_url_for_generic_adapter(self):
        with self.assertRaises(RuntimeError):
            build_adapter("http_json", Options(url=""))

    def test_html_escapes_answer_and_lists_failure(self):
        page = render_html(
            {"run_id": "run-1", "dataset": {}, "adapter": {}},
            [{
                "sample_id": "sample_0002",
                "success": False,
                "error_class": "http_server_error",
                "error": "HTTP 错误: 500",
                "question": "问题",
                "answer": "<script>alert(1)</script>",
                "ground_truth": "参考",
                "contexts": [],
                "latency_ms": 10,
            }],
            {"metrics": {"total": 1, "success_count": 0, "failure_count": 1}},
            {"summary": {}, "samples": [], "warnings": ["接口成功率不是答案正确率。"]},
        )
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>alert", page)
        self.assertIn("sample_0002", page)
        self.assertIn("HTTP 错误: 500", page)

    def test_show_writes_html_for_existing_result(self):
        from app.view import publish_existing_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "eval_demo.jsonl"
            result.write_text(json.dumps({
                "run_id": "demo",
                "sample_id": "sample_0001",
                "success": True,
                "question": "问题",
                "answer": "回答",
                "ground_truth": "回答",
                "contexts": ["片段"],
                "latency_ms": 5,
                "error": None,
                "error_class": "ok",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            path = publish_existing_run(result, root)
            text = path.read_text(encoding="utf-8")
            self.assertTrue(path.name == "report.html")
            self.assertIn("sample_0001", text)
            self.assertIn("片段", text)


if __name__ == "__main__":
    unittest.main()
