
"""
EvalHub 评测报告生成器。

读取已有评测结果，不重新调用 Agent，
计算基础指标并保存评测报告。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.metrics.basic_metrics import BasicMetrics


class ReportGenerator:
    """评测报告生成器。"""

    def load_results(self, file_path: str) -> list[dict[str, Any]]:
        """读取 JSONL 评测结果。"""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"评测结果不存在：{path}")

        results = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"第 {line_number} 行 JSON 格式错误"
                    ) from exc

                if not isinstance(item, dict):
                    raise ValueError(
                        f"第 {line_number} 行必须是 JSON 对象"
                    )

                results.append(item)

        if not results:
            raise ValueError("评测结果文件为空")

        return results

    def generate(
        self,
        input_path: str,
        output_dir: str = "results/reports",
        output_name: str = "evaluation_report.json",
    ) -> Path:
        """生成评测报告并保存。"""

        results = self.load_results(input_path)

        metrics = BasicMetrics.calculate(results)

        run_ids = {
            item.get("run_id")
            for item in results
            if item.get("run_id")
        }

        report = {
            "report_generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_file": str(input_path),
            "run_ids": sorted(run_ids),
            "metrics": metrics,
        }

        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        output_path = report_dir / output_name

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path
