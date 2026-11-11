import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.adapters.base import AgentRequest
from app.adapters.http_json import HttpJsonAdapter
from app.catalog import adapter_from_profile, get_agent, save_custom_agent
from app.web import create_app


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_answer_and_bearer_prefix(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "status": "success",
                "result": {"answer": "嵌套回答", "contexts": ["片段"]},
            })

        os.environ["EVALHUB_GATEWAY_TOKEN"] = "jwt-value"
        self.addCleanup(os.environ.pop, "EVALHUB_GATEWAY_TOKEN", None)
        adapter = HttpJsonAdapter(
            "http://example.test/dispatch",
            query_field="query",
            answer_field="result.answer",
            contexts_field="result.contexts",
            token_env="EVALHUB_GATEWAY_TOKEN",
            token_header="Authorization",
            token_prefix="Bearer ",
            extra_fields={"force_kind": "rag"},
            top_k=None,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            name="fire_gateway",
        )
        response = await adapter.invoke(AgentRequest(query="问题"))
        self.assertTrue(response.success)
        self.assertEqual(response.answer, "嵌套回答")
        self.assertEqual(response.contexts, ["片段"])
        self.assertEqual(seen["authorization"], "Bearer jwt-value")
        self.assertEqual(seen["body"]["force_kind"], "rag")
        self.assertNotIn("top_k", seen["body"])
        self.assertNotIn("jwt-value", json.dumps(response.raw_response))

    def test_saved_agent_rejects_raw_token_and_voice_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(ValueError):
                save_custom_agent(directory, {
                    "name": "other",
                    "url": "http://127.0.0.1:9000/ask",
                    "token_env": "raw-token",
                })
            saved = save_custom_agent(directory, {
                "name": "other_agent",
                "url": "http://127.0.0.1:9000/ask",
                "answer_field": "result.answer",
                "token_env": "OTHER_AGENT_TOKEN",
                "token_prefix": "Bearer",
            })
            text = (directory / "other_agent.json").read_text(encoding="utf-8")
            self.assertNotIn("secret", text)
            self.assertEqual(saved["token_prefix"], "Bearer ")
            loaded = get_agent(directory, "other_agent")
            adapter = adapter_from_profile(loaded, top_k=None)
            self.assertEqual(adapter.name, "other_agent")
            voice = get_agent(directory, "agent2_voice")
            with self.assertRaises(RuntimeError):
                adapter_from_profile(voice)

    def test_site_saves_custom_agent_without_calling_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(Path(tmp)))
            saved = client.post("/agents", data={
                "name": "lab_agent",
                "url": "http://127.0.0.1:9000/ask",
                "query_field": "question",
                "answer_field": "answer",
                "contexts_field": "contexts",
                "success_value": "success",
                "token_env": "LAB_TOKEN",
                "token_header": "X-Internal-Token",
                "token_prefix": "",
            }, follow_redirects=False)
            self.assertEqual(saved.status_code, 303)
            page = client.get("/agents")
            self.assertEqual(page.status_code, 200)
            self.assertIn("lab_agent", page.text)
            self.assertIn("语音交互", page.text)
            self.assertIn("登记其他评测对象", page.text)
            self.assertNotIn("LAB_TOKEN_VALUE", page.text)


if __name__ == "__main__":
    unittest.main()
