"""裁判评判：四轮轮换顺序 + 多数投票 + 平均分。"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

from arena.config import JudgeConfig
from arena.llm_client import LLMClient
from arena.prompts import JUDGE_EVAL_PROMPT, JUDGE_SELF_CHECK, format_answers_for_judge

logger = logging.getLogger(__name__)

LABELS = ["A", "B", "C", "D"]
# 四轮固定轮换顺序，消除位置偏见
ROTATION_ORDERS = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
]


@dataclass
class JudgeResult:
    winner_index: int
    winner_id: str
    winner_name: str
    winner_answer: str
    vote_counts: dict[str, int]
    average_scores: dict[str, float]
    round_details: list[dict]


class Judge:
    def __init__(self, config: JudgeConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    def _evaluate_once(
        self,
        question: str,
        answers: list[tuple[str, str, str]],
        order: list[int],
    ) -> dict:
        ordered = [answers[i] for i in order]
        labels = LABELS[: len(ordered)]
        answers_block = format_answers_for_judge(ordered, labels)

        prompt = JUDGE_EVAL_PROMPT.format(
            persona=self.config.persona + "\n" + JUDGE_SELF_CHECK,
            n=len(ordered),
            question=question,
            answers_block=answers_block,
        )
        messages = [
            {"role": "system", "content": f"你是{self.config.name}。"},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, model=self.llm.config.judge_model, temperature=0.3)
        return self._parse_judge_response(raw, labels, order)

    def _parse_judge_response(self, raw: str, labels: list[str], order: list[int]) -> dict:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning("裁判输出无法解析 JSON，默认选 A: %s", raw[:200])
            return {"winner_label": "A", "scores": {}, "raw": raw}

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("裁判 JSON 解析失败，默认选 A")
            return {"winner_label": "A", "scores": {}, "raw": raw}

        winner = str(data.get("winner", "A")).upper().strip()
        if winner not in labels:
            winner = labels[0]

        scores_by_label: dict[str, float] = {}
        for item in data.get("scores", []):
            label = str(item.get("candidate", "")).upper()
            total = item.get("total")
            if total is None:
                dims = ["accuracy", "depth", "logic", "terminology", "completeness"]
                total = sum(float(item.get(d, 0)) for d in dims)
            scores_by_label[label] = float(total)

        # 映射回原始专家索引
        label_to_original = {labels[i]: order[i] for i in range(len(order))}
        return {
            "winner_label": winner,
            "winner_original_index": label_to_original.get(winner, order[0]),
            "scores": scores_by_label,
            "raw": raw,
        }

    def evaluate(
        self,
        question: str,
        answers: list[tuple[str, str, str]],
    ) -> JudgeResult:
        """四轮轮换评判，多数投票 + 平均分决出胜者。"""
        if len(answers) != 4:
            raise ValueError(f"需要 4 个候选回答，实际 {len(answers)}")

        round_details: list[dict] = []
        score_accum: dict[int, list[float]] = {i: [] for i in range(4)}

        for round_idx, order in enumerate(ROTATION_ORDERS):
            detail = self._evaluate_once(question, answers, order)
            detail["rotation_round"] = round_idx + 1
            detail["order"] = order
            round_details.append(detail)

            for label, score in detail.get("scores", {}).items():
                if label in LABELS:
                    orig = order[LABELS.index(label)]
                    score_accum[orig].append(score)

        # 多数投票
        vote_for_index: list[int] = []
        for detail in round_details:
            vote_for_index.append(detail["winner_original_index"])
        vote_counter = Counter(vote_for_index)
        top_votes = vote_counter.most_common()
        winners_by_vote = [idx for idx, cnt in top_votes if cnt == top_votes[0][1]]

        # 平均分
        avg_scores = {
            i: (sum(scores) / len(scores) if scores else 0.0)
            for i, scores in score_accum.items()
        }

        if len(winners_by_vote) == 1:
            final_index = winners_by_vote[0]
        else:
            final_index = max(winners_by_vote, key=lambda i: avg_scores.get(i, 0))

        winner_id, winner_name, winner_answer = answers[final_index]
        vote_counts = {LABELS[i]: vote_counter.get(i, 0) for i in range(4)}
        average_scores = {LABELS[i]: round(avg_scores.get(i, 0), 2) for i in range(4)}

        return JudgeResult(
            winner_index=final_index,
            winner_id=winner_id,
            winner_name=winner_name,
            winner_answer=winner_answer,
            vote_counts=vote_counts,
            average_scores=average_scores,
            round_details=round_details,
        )
