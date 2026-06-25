"""单场竞技场：多轮专家辩论 → 评判 → 追问 → 循环。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from arena.config import AppConfig
from arena.expert import ExpertPanel
from arena.judge import Judge
from arena.llm_client import LLMClient
from arena.prompts import FOLLOWUP_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RoundRecord:
    round_num: int
    question: str
    expert_answers: list[dict]
    judge_result: dict
    winner_id: str
    winner_name: str
    winner_answer: str


@dataclass
class ArenaResult:
    arena_id: str
    initial_question: str
    rounds: list[RoundRecord] = field(default_factory=list)
    finetune_messages: list[dict[str, str]] = field(default_factory=list)


class ArenaSession:
    def __init__(self, config: AppConfig, llm: LLMClient, arena_id: str):
        self.config = config
        self.llm = llm
        self.arena_id = arena_id
        self.panel = ExpertPanel(
            config.experts, llm, max_workers=config.arena.max_concurrent_experts
        )
        self.judge = Judge(config.judge, llm)

    def _generate_followup(
        self,
        initial_question: str,
        round_num: int,
        winner_name: str,
        winner_answer: str,
    ) -> str:
        prompt = FOLLOWUP_PROMPT.format(
            initial_question=initial_question,
            round_num=round_num,
            winner_name=winner_name,
            winner_answer=winner_answer,
        )
        messages = [
            {"role": "system", "content": "你是技术圆桌主持人。"},
            {"role": "user", "content": prompt},
        ]
        return self.llm.chat(messages, model=self.llm.config.expert_model, temperature=0.8)

    def run(self, initial_question: str) -> ArenaResult:
        current_question = initial_question
        history: list[dict[str, str]] = []
        rounds: list[RoundRecord] = []
        finetune_messages: list[dict[str, str]] = []

        for round_num in range(1, self.config.arena.num_rounds + 1):
            logger.info("[%s] 第 %d 轮: %s", self.arena_id, round_num, current_question[:60])

            answers = self.panel.generate_all(current_question, history)
            judge_result = self.judge.evaluate(current_question, answers)

            expert_answers = [
                {"id": eid, "name": name, "content": content}
                for eid, name, content in answers
            ]

            record = RoundRecord(
                round_num=round_num,
                question=current_question,
                expert_answers=expert_answers,
                judge_result={
                    "vote_counts": judge_result.vote_counts,
                    "average_scores": judge_result.average_scores,
                    "round_details": judge_result.round_details,
                },
                winner_id=judge_result.winner_id,
                winner_name=judge_result.winner_name,
                winner_answer=judge_result.winner_answer,
            )
            rounds.append(record)

            finetune_messages.append({"role": "user", "content": current_question})
            finetune_messages.append(
                {"role": "assistant", "content": judge_result.winner_answer}
            )

            history = finetune_messages.copy()

            if round_num < self.config.arena.num_rounds:
                current_question = self._generate_followup(
                    initial_question,
                    round_num,
                    judge_result.winner_name,
                    judge_result.winner_answer,
                )

        return ArenaResult(
            arena_id=self.arena_id,
            initial_question=initial_question,
            rounds=rounds,
            finetune_messages=finetune_messages,
        )
