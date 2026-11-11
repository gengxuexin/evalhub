
from app.metrics.report import ReportGenerator


def main():
    generator = ReportGenerator()

    # 使用之前真实运行产生的结果文件
    input_path = (
        "results/eval_20261110_125549_68f4e5f9.jsonl"
    )

    output_path = generator.generate(input_path)

    print("\n========== EvalHub 评测报告 ==========")
    print("报告生成成功：", output_path)

    # 重新读取报告，验证保存结果
    import json

    with output_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    metrics = report["metrics"]

    print("样本总数：", metrics["total"])
    print("成功数：", metrics["success_count"])
    print("失败数：", metrics["failure_count"])
    print("成功率：", f"{metrics['success_rate']:.2%}")

    print(
        "平均耗时：",
        round(metrics["average_latency_ms"], 2),
        "ms",
    )

    print(
        "P95 耗时：",
        round(metrics["p95_latency_ms"], 2),
        "ms",
    )

    print("报告路径：", output_path)
    print("======================================")


if __name__ == "__main__":
    main()
