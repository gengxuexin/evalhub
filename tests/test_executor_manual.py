
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from app.adapters.fire_agent import FireAgentAdapter
from app.datasets.loader import DatasetLoader
from app.runner.executor import EvaluationExecutor
from app.storage.result_store import ResultStore

# 读取消防 Agent 的环境配置
# 当前目录结构：fire_agent_v2/
#                  ├── fire_agents/
#                  └── EvalHub/
FIRE_ENV_PATH = (
    Path(__file__).resolve().parents[2]
    / "fire_agents"
    / ".env"
)

if not FIRE_ENV_PATH.exists():
    raise FileNotFoundError(
        f"消防 Agent 配置文件不存在：{FIRE_ENV_PATH}"
    )

load_dotenv(FIRE_ENV_PATH)

# 检查 Token 是否成功加载，但不打印密钥
import os

if not os.getenv("AGENT_SHARED_TOKEN"):
    raise RuntimeError(
        "没有读取到 AGENT_SHARED_TOKEN，请检查消防项目 .env"
    )


async def main():
    # 1. 创建消防 Agent 适配器
    adapter = FireAgentAdapter(
        base_url="http://127.0.0.1:8001",
        top_k=5,
    )

    # 2. 加载评测数据
    loader = DatasetLoader()

    samples = loader.load_jsonl(
        "data/datasets/fire_qa.jsonl"
    )

    # 3. 创建执行器
    executor = EvaluationExecutor(adapter)

    # 4. 执行前 3 条样本
    results = await executor.run_batch(
        samples,
        limit=3,
    )
    # 保存本轮真实评测结果
    store = ResultStore(output_dir="results")

    output_path = store.save_results(results)

    print("\n========== 结果保存 ==========")
    print("本轮结果文件：", output_path)
    print("保存记录数：", len(results))
    print("==============================")

    print("\n第一条执行结果：")
    print("问题：", results[0]["question"])
    print("回答：", results[0]["answer"][:300])
    print("标准答案：", results[0]["ground_truth"])
    print("成功：", results[0]["success"])


if __name__ == "__main__":
    asyncio.run(main())
