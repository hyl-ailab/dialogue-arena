"""OpenAI 兼容 API 客户端封装。"""

from __future__ import annotations

import logging
from typing import Sequence

from openai import OpenAI

from arena.config import LLMConfig
from arena.utils import clean_model_output

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model = model or self.config.expert_model
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return clean_model_output(content)
        except Exception as exc:
            logger.error("LLM 调用失败 model=%s: %s", model, exc)
            raise
