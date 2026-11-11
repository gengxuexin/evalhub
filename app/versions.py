"""评测指标套件版本。用于说明某次评分使用的提示词，而不是模型回答版本。"""

import hashlib

from app.metrics.judge import (
    FAITHFULNESS_SYSTEM,
    RECALL_SYSTEM,
    RELEVANCE_RATIO_SYSTEM,
    RELEVANCY_SYSTEM,
)

SUITE_VERSION = "2026-11-11.2"


def suite_fingerprint() -> dict:
    prompts = "\n".join([
        FAITHFULNESS_SYSTEM,
        RELEVANCY_SYSTEM,
        RELEVANCE_RATIO_SYSTEM,
        RECALL_SYSTEM,
    ])
    return {
        "suite_version": SUITE_VERSION,
        "judge_prompt_sha256": hashlib.sha256(prompts.encode("utf-8")).hexdigest(),
    }
