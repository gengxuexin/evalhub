
"""
EvalHub 基础评测指标。

计算：
1. 请求成功率
2. 请求失败率
3. 平均响应时间
4. 中位响应时间
5. P95 响应时间
6. 最快和最慢响应时间
"""

import math
from typing import Any


class BasicMetrics:
    """基础评测指标计算器。"""

    @staticmethod
    def calculate(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """根据批量执行结果计算基础指标。"""

        if not results:
            raise ValueError("评测结果不能为空")

        total = len(results)

        success_count = sum(
            1 for item in results
            if item.get("success") is True
        )

        failure_count = total - success_count

        # 只统计有效的非负响应耗时
        latencies = []

        for item in results:
            value = item.get("latency_ms")

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            ):
                latencies.append(float(value))

        if not latencies:
            average_latency = None
            median_latency = None
            p95_latency = None
            min_latency = None
            max_latency = None
        else:
            sorted_latencies = sorted(latencies)

            average_latency = sum(latencies) / len(latencies)

            n = len(sorted_latencies)

            if n % 2 == 1:
                median_latency = sorted_latencies[n // 2]
            else:
                median_latency = (
                    sorted_latencies[n // 2 - 1]
                    + sorted_latencies[n // 2]
                ) / 2

            # P95 使用线性插值法
            position = (n - 1) * 0.95
            lower_index = math.floor(position)
            upper_index = math.ceil(position)

            lower_value = sorted_latencies[lower_index]
            upper_value = sorted_latencies[upper_index]

            p95_latency = (
                lower_value
                + (upper_value - lower_value)
                * (position - lower_index)
            )

            min_latency = sorted_latencies[0]
            max_latency = sorted_latencies[-1]

        return {
            "total": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / total,
            "failure_rate": failure_count / total,
            "valid_latency_count": len(latencies),
            "average_latency_ms": average_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
        }
