
"""
EvalHub 批量评测执行器。

职责：
1. 接收数据集样本
2. 调用统一 AgentAdapter
3. 收集每条样本的执行结果
4. 保留标准答案，供后续质量评分使用
"""

from datetime import datetime, timezone
from typing import Sequence

from app.adapters.base import AgentAdapter, AgentRequest
from app.datasets.loader import DatasetSample


class EvaluationExecutor:
    """评测任务执行器。"""

    def __init__(self, adapter: AgentAdapter):
        self.adapter = adapter

    async def run_sample(
        self,
        sample: DatasetSample,
    ) -> dict:
        """执行单条评测样本。"""

        request = AgentRequest(
            query=sample.question,
            sample_id=sample.sample_id,
            metadata={
                "category": sample.category,
                "difficulty": sample.difficulty,
                "verified": sample.verified,
            },
        )

        started_at = datetime.now(timezone.utc)

        try:
            response = await self.adapter.invoke(request)

            return {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "verified": sample.verified,

                "answer": response.answer,
                "success": response.success,
                "latency_ms": response.latency_ms,
                "status_code": response.status_code,
                "error": response.error,
                "model": response.model,
                "contexts": response.contexts,
                "raw_response": response.raw_response,

                "started_at": started_at.isoformat(),
            }

        except Exception as exc:
            # 单条样本失败，不影响整个批次
            return {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "verified": sample.verified,

                "answer": "",
                "success": False,
                "latency_ms": 0.0,
                "status_code": None,
                "error": f"{type(exc).__name__}: {exc}",
                "model": None,
                "contexts": [],
                "raw_response": {},

                "started_at": started_at.isoformat(),
            }

    async def run_batch(
        self,
        samples: Sequence[DatasetSample],
        limit: int | None = None,
    ) -> list[dict]:
        """批量执行评测样本。"""

        if limit is not None:
            if limit <= 0:
                raise ValueError("limit 必须大于 0")
            samples = samples[:limit]

        results = []

        total = len(samples)

        print(f"\n开始执行评测，共 {total} 条样本")
        print("=" * 45)

        for index, sample in enumerate(samples, start=1):
            print(
                f"[{index}/{total}] "
                f"正在执行：{sample.sample_id}"
            )

            result = await self.run_sample(sample)
            results.append(result)

            status = "成功" if result["success"] else "失败"

            print(
                f"  状态：{status} | "
                f"耗时：{result['latency_ms']:.2f} ms"
            )

            if result["error"]:
                print(f"  错误：{result['error']}")

        success_count = sum(
            1 for item in results if item["success"]
        )

        print("=" * 45)
        print(f"执行完成：{len(results)} 条")
        print(f"成功：{success_count} 条")
        print(f"失败：{len(results) - success_count} 条")

        return results
