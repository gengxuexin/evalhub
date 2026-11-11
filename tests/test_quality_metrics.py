import unittest

from app.metrics.judge import (
    FAITHFULNESS_SYSTEM,
    RECALL_SYSTEM,
    answer_relevancy,
    context_recall,
    context_relevance_ratio,
    faithfulness,
)
from app.metrics.lexical import character_f1, normalized_exact_match
from app.metrics.quality import score_sample, summarize_quality
from app.metrics.semantic import semantic_similarity


class ScriptedJudge:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.model_name = "scripted-test"
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedEmbedder:
    def __init__(self, vectors=None, error=None):
        self.model_name = "embed-test"
        self.vectors = vectors
        self.error = error

    def embed(self, texts):
        if self.error:
            raise self.error
        return self.vectors


def _record(**overrides):
    record = {
        "sample_id": "sample_0001",
        "success": True,
        "question": "公众聚集场所投入使用营业前消防安全检查实行什么制度？",
        "answer": "实行告知承诺管理。",
        "ground_truth": "实行告知承诺管理，申请人可自主选择采用告知承诺方式办理",
        "contexts": ["实行告知承诺管理。"],
        "raw_response": {"status": "success", "contexts": ["实行告知承诺管理。"]},
    }
    record.update(overrides)
    return record


class LexicalTests(unittest.TestCase):
    def test_exact_match_ignores_punctuation(self):
        result = normalized_exact_match(
            _record(
                answer="《消防法》第六十条",
                ground_truth="消防法第六十条",
            )
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(result.score, 1.0)
        self.assertFalse(result.detail["suitable_as_primary_quality_score"])

    def test_longer_correct_answer_is_not_exact(self):
        result = normalized_exact_match(_record())
        self.assertEqual(result.score, 0.0)

    def test_character_f1_is_overlap_not_semantic_label(self):
        result = character_f1(
            _record(
                answer="实行告知承诺",
                ground_truth="实行告知承诺管理",
            )
        )
        self.assertEqual(result.status, "scored")
        self.assertAlmostEqual(result.score, 12 / 14)
        self.assertFalse(result.detail["equals_semantic_correctness"])

    def test_failed_agent_call_is_not_zero(self):
        result = normalized_exact_match(_record(success=False, answer=""))
        self.assertEqual(result.status, "not_evaluable")
        self.assertIsNone(result.score)
        self.assertEqual(result.detail["reason"], "agent_call_failed")


class JudgeTests(unittest.TestCase):
    def test_missing_contexts_does_not_call_judge_or_score_zero(self):
        judge = ScriptedJudge(['{"claims":["x"],"verdicts":[{"claim_index":0,"verdict":"supported"}]}'])
        result = faithfulness(_record(contexts=[], raw_response={"status": "success"}), judge)
        self.assertEqual(result.status, "not_evaluable")
        self.assertIsNone(result.score)
        self.assertEqual(result.detail["reason"], "empty_contexts")
        self.assertEqual(judge.calls, [])

    def test_contexts_absent_from_api_are_not_evaluable(self):
        judge = ScriptedJudge([])
        result = faithfulness(
            _record(contexts=None, raw_response={"status": "success", "answer": "x"}),
            judge,
        )
        self.assertEqual(result.detail["reason"], "contexts_not_returned")
        self.assertEqual(judge.calls, [])

    def test_verdict_count_mismatch_is_invalid(self):
        judge = ScriptedJudge([
            '{"claims":["甲","乙"],"verdicts":[{"claim_index":0,"verdict":"supported","reason":"ok"}]}'
        ])
        result = faithfulness(_record(), judge)
        self.assertEqual(result.status, "invalid_output")
        self.assertIsNone(result.score)
        self.assertEqual(result.detail["reason"], "verdict_count_mismatch")

    def test_bad_json_is_invalid_output(self):
        result = faithfulness(_record(), ScriptedJudge(["这不是 JSON"]))
        self.assertEqual(result.status, "invalid_output")
        self.assertIsNone(result.score)

    def test_judge_exception_is_not_zero(self):
        result = faithfulness(_record(), ScriptedJudge([RuntimeError("timeout")]))
        self.assertEqual(result.status, "judge_failure")
        self.assertIsNone(result.score)
        self.assertIn("timeout", result.error)

    def test_all_insufficient_abstains(self):
        raw = (
            '{"claims":["甲"],"verdicts":[{"claim_index":0,'
            '"verdict":"insufficient","reason":"上下文没有该句"}]}'
        )
        result = faithfulness(_record(), ScriptedJudge([raw]))
        self.assertEqual(result.status, "judge_abstain")
        self.assertIsNone(result.score)

    def test_faithfulness_excludes_insufficient_from_denominator(self):
        raw = (
            '{"claims":["甲","乙","丙"],"verdicts":['
            '{"claim_index":0,"verdict":"supported","reason":"1"},'
            '{"claim_index":1,"verdict":"unsupported","reason":"无"},'
            '{"claim_index":2,"verdict":"insufficient","reason":"不够"}]}'
        )
        result = faithfulness(_record(), ScriptedJudge([raw]))
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.detail["insufficient"], 1)
        self.assertFalse(result.detail["used_second_retrieval"])

    def test_relevancy_uses_three_levels_not_division_by_five(self):
        result = answer_relevancy(
            _record(),
            ScriptedJudge(['{"label":"partial","reason":"只回答了制度名"}']),
        )
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.detail["label"], "partial")
        self.assertFalse(result.detail["divided_by_generated_question_count"])
        self.assertFalse(result.detail["is_calibrated_probability"])

    def test_average_precision_uses_one_judge_call_and_original_order(self):
        from app.metrics.judge import retrieval_context_metrics

        record = _record(contexts=["甲", "乙", "丙", "丁"])
        raw = (
            '{"verdicts":['
            '{"context_index":0,"relevant":true,"reason":"a"},'
            '{"context_index":1,"relevant":false,"reason":"b"},'
            '{"context_index":2,"relevant":true,"reason":"c"},'
            '{"context_index":3,"relevant":false,"reason":"d"}]}'
        )
        judge = ScriptedJudge([raw])
        metrics = retrieval_context_metrics(record, judge)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(metrics["context_relevance_ratio"].score, 0.5)
        self.assertAlmostEqual(metrics["context_average_precision"].score, (1 + 2 / 3) / 2)
        self.assertTrue(metrics["context_average_precision"].detail["considers_rank"])
        self.assertFalse(metrics["context_relevance_ratio"].detail["considers_rank"])
        record = _record(contexts=["甲", "乙", "丙", "丁"])
        raw = (
            '{"verdicts":['
            '{"context_index":0,"relevant":true,"reason":"a"},'
            '{"context_index":1,"relevant":false,"reason":"b"},'
            '{"context_index":2,"relevant":true,"reason":"c"},'
            '{"context_index":3,"relevant":false,"reason":"d"}]}'
        )
        result = context_relevance_ratio(record, ScriptedJudge([raw]))
        self.assertEqual(result.name, "context_relevance_ratio")
        self.assertEqual(result.score, 0.5)
        self.assertFalse(result.detail["considers_rank"])
        self.assertFalse(result.detail["metric_is_rank_aware_context_precision"])

    def test_relevance_ratio_rejects_short_verdict_list(self):
        record = _record(contexts=["甲", "乙"])
        raw = '{"verdicts":[{"context_index":0,"relevant":true,"reason":"a"}]}'
        result = context_relevance_ratio(record, ScriptedJudge([raw]))
        self.assertEqual(result.status, "invalid_output")
        self.assertIsNone(result.score)

    def test_recall_prompt_rejects_topic_overlap(self):
        self.assertIn("主题相关不等于覆盖", RECALL_SYSTEM)
        self.assertIn("主题相关", FAITHFULNESS_SYSTEM)
        judge = ScriptedJudge([
            '{"facts":["告知承诺","可自主选择"],"verdicts":['
            '{"fact_index":0,"verdict":"covered","reason":"原文"},'
            '{"fact_index":1,"verdict":"not_covered","reason":"只是同一部法律"}]}'
        ])
        result = context_recall(_record(), judge)
        self.assertEqual(result.score, 0.5)
        self.assertFalse(result.detail["topic_overlap_counts_as_coverage"])
        self.assertIn("主题相关不等于覆盖", judge.calls[0][1])

    def test_summary_skips_unscored_items(self):
        samples = [
            score_sample(_record(answer="实行告知承诺管理，申请人可自主选择采用告知承诺方式办理")),
            score_sample(_record(success=False, answer="")),
        ]
        summary = summarize_quality(samples)
        exact = summary["normalized_exact_match"]
        self.assertEqual(exact["scored_count"], 1)
        self.assertEqual(exact["mean_score"], 1.0)
        self.assertEqual(exact["status_counts"]["not_evaluable"], 1)


class SemanticTests(unittest.TestCase):
    def test_cosine_is_reported_without_threshold(self):
        embedder = ScriptedEmbedder(vectors=[[1.0, 0.0], [0.0, 1.0]])
        result = semantic_similarity(_record(), embedder)
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(result.detail["pass_threshold"])
        self.assertEqual(result.detail["model"], "embed-test")
        self.assertFalse(result.detail["rescaled_to_probability"])

    def test_embedder_error_is_not_zero(self):
        result = semantic_similarity(
            _record(),
            ScriptedEmbedder(error=RuntimeError("embed down")),
        )
        self.assertEqual(result.status, "metric_failure")
        self.assertIsNone(result.score)


if __name__ == "__main__":
    unittest.main()
