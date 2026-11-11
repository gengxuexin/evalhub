"""真实 Judge 与 Embedding 客户端。

密钥只从已加载的环境或消防项目配置读取，不写入日志。
"""

import httpx
from openai import OpenAI


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 90.0,
    ):
        if not api_key:
            raise RuntimeError("Judge API key 为空")
        self.model_name = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Judge 返回空内容")
        return content


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise RuntimeError("Embedding API key 为空")
        self.model_name = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise RuntimeError("Embedding 返回数量与输入不一致")
        return [list(item.embedding) for item in ordered]


def build_live_clients(fire_agents_root: str):
    """使用消防项目的配置对象创建客户端，避免在 EvalHub 里复制密钥。"""

    import sys

    if fire_agents_root not in sys.path:
        sys.path.insert(0, fire_agents_root)

    from common.config import embedding_settings, get_settings

    settings = get_settings()
    embedding = embedding_settings()
    judge = OpenAICompatibleJudge(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.judge_model,
    )
    embedder = OpenAICompatibleEmbedder(
        api_key=embedding["api_key"],
        base_url=embedding["base_url"],
        model=embedding["model"],
    )
    return judge, embedder
