# Dialogue Arena

<p align="center">
  <img src="docs/architecture.svg" alt="Architecture" width="720"/>
</p>

<p align="center">
  <b>Self-Evolving Multi-Model Multi-Turn Dialogue Corpus Generation Arena</b>
</p>

<p align="center">
  <a href="README.md">中文</a>
</p>

---

High-quality training data is the key to LLM capability ceilings. Manual annotation is expensive, slow, and inconsistent.

**Dialogue Arena** orchestrates multi-expert AI debates with automatic judging and iterative follow-ups to produce SFT-ready multi-turn dialogue corpora — fully automated.

## Features

- **4 expert personas** answer in parallel each round
- **Independent judge** with 4-order rotation, majority vote, and average scores
- **Multi-round follow-ups** deepen the conversation automatically
- **Concurrent arenas** via `ThreadPoolExecutor`
- **Self-check prompts** for accuracy, depth, logic, and role consistency
- **Output cleaning** removes internal thinking tags
- **FileLock** for safe concurrent JSONL writes
- **Timestamped experiment dirs** for reproducibility

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY and OPENAI_BASE_URL

python main.py
python main.py --question "What are the core challenges of 6G network slicing?" --rounds 3
```

## Output

`final_answers_finetune_multiturn.jsonl` — standard `messages` format for Llama, Qwen, ChatGLM SFT.

## License

MIT
