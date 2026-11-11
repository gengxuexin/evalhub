
"""
EvalHub 评测结果存储模块。

将批量评测结果保存为 JSONL 文件。
每次运行生成独立文件，避免覆盖历史结果。
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResultStore:
    """评测结果持久化管理。"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        results: list[dict[str, Any]],
    ) -> Path:
        """保存一批评测结果，返回生成的文件路径。"""

        if not results:
            raise ValueError("没有评测结果可以保存")

        now = datetime.now(timezone.utc)
        run_id = (
            now.strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )

        output_path = self.output_dir / f"eval_{run_id}.jsonl"
        temp_path = output_path.with_suffix(".tmp")

        try:
            with temp_path.open("w", encoding="utf-8") as file:
                for result in results:
                    record = {
                        **result,
                        "run_id": run_id,
                    }

                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

            # 文件完整写入后再替换为正式结果文件
            temp_path.replace(output_path)

        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return output_path
