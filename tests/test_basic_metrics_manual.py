
from app.metrics.basic_metrics import BasicMetrics


def main():
    # 使用模拟数据测试计算逻辑，不调用真实模型
    results = [
        {"success": True, "latency_ms": 1000},
        {"success": True, "latency_ms": 2000},
        {"success": False, "latency_ms": 3000},
        {"success": True, "latency_ms": 4000},
        {"success": True, "latency_ms": 5000},
    ]

    metrics = BasicMetrics.calculate(results)

    print("\n========== 基础指标测试 ==========")

    print("总请求数：", metrics["total"])
    print("成功数：", metrics["success_count"])
    print("失败数：", metrics["failure_count"])

    print("成功率：", f"{metrics['success_rate']:.2%}")
    print("失败率：", f"{metrics['failure_rate']:.2%}")

    print(
        "平均耗时：",
        round(metrics["average_latency_ms"], 2),
        "ms",
    )

    print(
        "中位耗时：",
        round(metrics["median_latency_ms"], 2),
        "ms",
    )

    print(
        "P95 耗时：",
        round(metrics["p95_latency_ms"], 2),
        "ms",
    )

    print("最快耗时：", metrics["min_latency_ms"], "ms")
    print("最慢耗时：", metrics["max_latency_ms"], "ms")

    print("==================================")


if __name__ == "__main__":
    main()
