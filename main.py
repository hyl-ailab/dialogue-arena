"""CLI 入口。"""

from __future__ import annotations

import argparse
import logging
import sys

from arena.config import load_config
from arena.runner import ArenaRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="自进化多模型多轮对话语料生成竞技场",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py
  python main.py --question "6G网络切片技术面临哪些核心挑战？"
  python main.py --rounds 3 --config config/config.yaml
        """,
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--question", action="append", dest="questions", help="自定义初始问题（可多次指定）")
    parser.add_argument("--rounds", type=int, default=None, help="辩论轮数")
    parser.add_argument("--concurrency", type=int, default=None, help="并发竞技场数量")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    if args.rounds is not None:
        config.arena.num_rounds = args.rounds
    if args.concurrency is not None:
        config.arena.max_concurrent_arenas = args.concurrency

    if not config.llm.api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量或在 .env 中配置", file=sys.stderr)
        return 1

    questions = args.questions if args.questions else None
    runner = ArenaRunner(config)
    exp_dir = runner.run_all(questions)
    print(f"\n实验完成，输出目录:\n  {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
