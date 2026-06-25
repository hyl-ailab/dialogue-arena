"""并发运行多场竞技场。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from arena.arena import ArenaSession
from arena.config import AppConfig
from arena.llm_client import LLMClient
from arena.utils import append_jsonl, build_finetune_record, create_experiment_dir, write_json

logger = logging.getLogger(__name__)


class ArenaRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.exp_dir = create_experiment_dir(
            config.project_root / config.output.experiments_dir
        )
        self.finetune_path = self.exp_dir / config.output.finetune_filename
        self.raw_log_path = self.exp_dir / config.output.raw_log_filename
        self.meta_path = self.exp_dir / "experiment_meta.json"

        write_json(
            self.meta_path,
            {
                "experiment_dir": str(self.exp_dir),
                "num_rounds": config.arena.num_rounds,
                "num_questions": len(config.seed_questions),
                "experts": [{"id": e.id, "name": e.name} for e in config.experts],
                "expert_model": config.llm.expert_model,
                "judge_model": config.llm.judge_model,
            },
        )
        logger.info("实验目录: %s", self.exp_dir)

    def _run_one(self, idx: int, question: str) -> dict:
        arena_id = f"arena_{idx:04d}"
        session = ArenaSession(self.config, self.llm, arena_id)
        result = session.run(question)

        finetune_record = build_finetune_record(result.finetune_messages)
        append_jsonl(self.finetune_path, finetune_record)

        raw_record = {
            "arena_id": result.arena_id,
            "initial_question": result.initial_question,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "question": r.question,
                    "expert_answers": r.expert_answers,
                    "judge_result": r.judge_result,
                    "winner_id": r.winner_id,
                    "winner_name": r.winner_name,
                    "winner_answer": r.winner_answer,
                }
                for r in result.rounds
            ],
        }
        append_jsonl(self.raw_log_path, raw_record)
        return {"arena_id": arena_id, "question": question, "rounds": len(result.rounds)}

    def run_all(self, questions: list[str] | None = None) -> Path:
        questions = questions or self.config.seed_questions
        if not questions:
            raise ValueError("未提供 seed_questions，请在 config.yaml 中配置或通过 CLI 传入")

        max_workers = min(self.config.arena.max_concurrent_arenas, len(questions))
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_one, i, q): (i, q)
                for i, q in enumerate(questions, start=1)
            }
            with tqdm(total=len(futures), desc="竞技场进度", unit="场") as pbar:
                for future in as_completed(futures):
                    i, q = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        logger.error("竞技场 arena_%04d 失败: %s — %s", i, q[:40], exc)
                        results.append({"arena_id": f"arena_{i:04d}", "error": str(exc)})
                    pbar.update(1)

        summary_path = self.exp_dir / "summary.json"
        write_json(
            summary_path,
            {
                "total": len(questions),
                "success": sum(1 for r in results if "error" not in r),
                "failed": sum(1 for r in results if "error" in r),
                "finetune_output": str(self.finetune_path),
                "raw_log": str(self.raw_log_path),
                "results": results,
            },
        )
        logger.info("完成! 微调数据: %s", self.finetune_path)
        return self.exp_dir
