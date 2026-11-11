
import asyncio

from app.adapters.base import AgentRequest
from app.adapters.fire_agent import FireAgentAdapter
from dotenv import load_dotenv

# 读取原消防 Agent 的配置文件
load_dotenv(
    "/home/geng/fire_agent_v2/fire_agents/.env",
    override=True,
)

async def main():
    adapter = FireAgentAdapter(
        base_url="http://127.0.0.1:8001",
        top_k=5,
    )

    request = AgentRequest(
        query="消防法第六十条规定了什么？",
        sample_id="fire_test_001",
    )

    response = await adapter.invoke(request)

    print("\n========== EvalHub 测试结果 ==========")
    print("Agent:", adapter.name)
    print("成功:", response.success)
    print("HTTP 状态码:", response.status_code)
    print("耗时:", round(response.latency_ms, 2), "ms")
    print("错误:", response.error)
    print("回答:", response.answer[:500])
    print("======================================")


if __name__ == "__main__":
    asyncio.run(main())
