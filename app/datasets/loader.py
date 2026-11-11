
"""
EvalHub 数据集加载器。

负责：
1. 读取 JSONL 格式的数据集
2. 校验数据字段
3. 将原始 JSON 转换为统一的 DatasetSample 对象
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetSample:
    """统一的评测样本结构。"""

    sample_id: str
    question: str
    ground_truth: str
    category: str
    difficulty: str
    verified: bool
    metadata: dict[str, Any]


class DatasetLoader:
    """JSONL 数据集加载器。"""

    REQUIRED_FIELDS = [
        "question",
        "ground_truth",
        "category",
        "difficulty",
        "verified",
    ]

    def load_jsonl(self, file_path: str) -> list[DatasetSample]:
        """
        加载 JSONL 文件。

        Args:
            file_path: JSONL 文件路径

        Returns:
            DatasetSample 对象列表
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"数据集不存在：{path}")

        if not path.is_file():
            raise ValueError(f"指定路径不是文件：{path}")

        samples = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                # 跳过空行
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"第 {line_number} 行 JSON 格式错误：{exc}"
                    ) from exc

                if not isinstance(data, dict):
                    raise ValueError(
                        f"第 {line_number} 行必须是 JSON 对象"
                    )

                # 检查必需字段
                missing = [
                    field
                    for field in self.REQUIRED_FIELDS
                    if field not in data
                ]

                if missing:
                    raise ValueError(
                        f"第 {line_number} 行缺少字段：{missing}"
                    )

                # 检查字段内容
                for field in [
                    "question",
                    "ground_truth",
                    "category",
                    "difficulty",
                ]:
                    if not isinstance(data[field], str) or not data[field].strip():
                        raise ValueError(
                            f"第 {line_number} 行字段 {field} 必须是非空字符串"
                        )

                if not isinstance(data["verified"], bool):
                    raise ValueError(
                        f"第 {line_number} 行字段 verified 必须是布尔值"
                    )

                # 生成稳定的样本 ID
                sample_id = f"sample_{line_number:04d}"

                sample = DatasetSample(
                    sample_id=sample_id,
                    question=data["question"].strip(),
                    ground_truth=data["ground_truth"].strip(),
                    category=data["category"].strip(),
                    difficulty=data["difficulty"].strip(),
                    verified=data["verified"],
                    metadata={
                        key: value
                        for key, value in data.items()
                        if key not in self.REQUIRED_FIELDS
                    },
                )

                samples.append(sample)

        if not samples:
            raise ValueError("数据集为空，没有可用的评测样本")

        return samples
