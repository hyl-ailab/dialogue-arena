"""配置加载与环境变量解析。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ExpertConfig:
    id: str
    name: str
    persona: str


@dataclass
class JudgeConfig:
    name: str
    persona: str


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    expert_model: str
    judge_model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120


@dataclass
class ArenaConfig:
    num_rounds: int = 5
    max_concurrent_arenas: int = 8
    max_concurrent_experts: int = 4


@dataclass
class OutputConfig:
    experiments_dir: str = "experiments"
    finetune_filename: str = "final_answers_finetune_multiturn.jsonl"
    raw_log_filename: str = "arena_raw_log.jsonl"


@dataclass
class AppConfig:
    llm: LLMConfig
    arena: ArenaConfig
    experts: list[ExpertConfig]
    judge: JudgeConfig
    seed_questions: list[str]
    output: OutputConfig
    project_root: Path = field(default_factory=lambda: Path.cwd())


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _ENV_PATTERN.fullmatch(value.strip())
    if match:
        return os.environ.get(match.group(1), "")
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _walk_resolve(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(v) for v in obj]
    return _resolve_env(obj)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    load_dotenv()
    root = Path(__file__).resolve().parent.parent
    path = Path(config_path) if config_path else root / "config" / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = _walk_resolve(raw)

    llm_raw = data["llm"]
    llm = LLMConfig(
        api_key=llm_raw.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
        base_url=llm_raw.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        expert_model=os.environ.get("EXPERT_MODEL") or llm_raw["expert_model"],
        judge_model=os.environ.get("JUDGE_MODEL") or llm_raw["judge_model"],
        temperature=float(llm_raw.get("temperature", 0.7)),
        max_tokens=int(llm_raw.get("max_tokens", 2048)),
        timeout=int(llm_raw.get("timeout", 120)),
    )

    experts = [
        ExpertConfig(id=e["id"], name=e["name"], persona=e["persona"].strip())
        for e in data["experts"]
    ]
    judge = JudgeConfig(
        name=data["judge"]["name"],
        persona=data["judge"]["persona"].strip(),
    )
    arena = ArenaConfig(**data.get("arena", {}))
    output = OutputConfig(**data.get("output", {}))

    return AppConfig(
        llm=llm,
        arena=arena,
        experts=experts,
        judge=judge,
        seed_questions=data.get("seed_questions", []),
        output=output,
        project_root=root,
    )
