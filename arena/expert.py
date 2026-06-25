"""专家回答生成。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from arena.config import ExpertConfig
from arena.llm_client import LLMClient
from arena.prompts import EXPERT_ANSWER_PROMPT, SELF_CHECK_CRITERIA, format_history

logger = logging.getLogger(__name__)


class ExpertPanel:
    def __init__(self, experts: list[ExpertConfig], llm: LLMClient, max_workers: int = 4):
        self.experts = experts
        self.llm = llm
        self.max_workers = max_workers

    def _generate_one(
        self,
        expert: ExpertConfig,
        question: str,
        history: list[dict[str, str]],
    ) -> tuple[str, str, str]:
        prompt = EXPERT_ANSWER_PROMPT.format(
            persona=expert.persona,
            self_check=SELF_CHECK_CRITERIA,
            question=question,
            history=format_history(history),
        )
        messages = [
            {"role": "system", "content": f"你是{expert.name}。"},
            {"role": "user", "content": prompt},
        ]
        answer = self.llm.chat(messages, model=self.llm.config.expert_model)
        return expert.id, expert.name, answer

    def generate_all(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[tuple[str, str, str]]:
        """并行生成四位专家的回答。返回 [(id, name, answer), ...]"""
        history = history or []
        results: list[tuple[str, str, str]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._generate_one, expert, question, history): expert
                for expert in self.experts
            }
            for future in as_completed(futures):
                expert = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("专家 %s 生成失败: %s", expert.name, exc)
                    results.append((expert.id, expert.name, f"[生成失败: {exc}]"))

        # 保持与配置顺序一致，便于日志阅读
        order = {e.id: i for i, e in enumerate(self.experts)}
        results.sort(key=lambda x: order.get(x[0], 99))
        return results
