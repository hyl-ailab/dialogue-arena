# Dialogue Arena

<p align="center">
  <img src="docs/architecture.svg" alt="系统架构图" width="720"/>
</p>

<p align="center">
  <b>自进化多模型多轮对话语料生成竞技场</b><br/>
  Self-Evolving Multi-Model Multi-Turn Dialogue Corpus Generation Arena
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#核心思想">核心思想</a> ·
  <a href="#技术亮点">技术亮点</a> ·
  <a href="#输出格式">输出格式</a> ·
  <a href="README_EN.md">English</a>
</p>

---

在大模型时代，**高质量训练数据**是决定模型能力上限的关键。然而，人工标注成本高昂、效率低下，且难以保证专业性和一致性。

**Dialogue Arena** 是一套自进化多模型多轮对话语料生成系统。它像一场永不落幕的「AI 辩论赛」——让多个专业 AI 角色围绕同一话题展开多轮深度对话，通过自动评判筛选最优回答，最终生成可用于微调大模型的高质量多轮对话数据。

## 核心思想：让 AI 自己「卷」起来

传统数据生成是「单打独斗」——给一个模型一个问题，让它回答。本系统采用 **竞技场（Arena）** 模式，灵感来自 [Chatbot Arena](https://lmarena.ai/)：

| 步骤 | 说明 |
|------|------|
| 多角色对抗 | 4 位不同专业背景的 AI 专家同时回答同一问题 |
| 自动裁判 | 独立裁判模型从多维度评估，选出最优回答 |
| 多轮追问 | 基于最优回答自动生成深度追问，推动对话深入 |
| 循环迭代 | 重复上述过程，形成逻辑连贯的多轮对话 |
| 数据沉淀 | 保存完整辩论过程与最优回答链，输出 SFT 微调数据 |

**全程无需人工干预**，实现数据的自进化生成。

## 系统如何工作？

想象四位通信领域顶级专家围坐一桌讨论技术难题：

1. **主持人提问** — 系统提出初始问题，如「6G 网络切片技术面临哪些核心挑战？」
2. **专家各抒己见** — ITU 主席、3GPP 主席、IEEE 院士、IETF 委员同时发表专业见解
3. **评委打分** — 中立评委从技术准确性、深度、逻辑性等维度评判，选出本轮最佳
4. **追问挑战** — 根据最佳回答生成更深入追问，如「切片隔离机制在异构硬件下如何保证端到端性能？」
5. **循环往复** — 重复 5~8 轮，串联每轮最佳回答，形成层层递进的专业对话

## 技术亮点

- **大规模并发** — `ThreadPoolExecutor` 并行运行多场辩论与四专家回答，支持数十场同时执行
- **自我审查机制** — 专家与裁判生成前执行准确性、完整性、逻辑性、专业深度、角色一致性检查
- **消除评判偏见** — 四轮固定顺序轮换 + 多数投票 + 平均分，避免位置偏见
- **数据清洗** — 自动移除 ``、`<think>` 等内部思考标签
- **安全写入** — `FileLock` 保护 JSONL 并发写入，防止数据损坏
- **实验可追溯** — 每次运行创建时间戳实验目录，保存原始日志与微调数据

## 输出格式

最终微调数据 `final_answers_finetune_multiturn.jsonl`：

```json
{
  "messages": [
    {"role": "user", "content": "6G网络切片技术面临哪些核心挑战？"},
    {"role": "assistant", "content": "作为ITU-T首席科学家，我认为..."},
    {"role": "user", "content": "您提到的切片隔离机制，在异构硬件环境下如何保证端到端性能？"},
    {"role": "assistant", "content": "这是一个关键问题。根据Y.3102建议书..."}
  ]
}
```

兼容 Llama、Qwen、ChatGLM 等主流模型的指令微调（SFT）格式。参见 [`examples/sample_output.jsonl`](examples/sample_output.jsonl)。

## 应用场景

- **大模型微调** — 为通信、科技等垂直领域生成高质量领域对话数据
- **智能客服训练** — 模拟专家与用户的多轮深度问答
- **知识库构建** — 自动化生成结构化专业知识问答对
- **AI 能力评测** — 作为基准平台，评估不同模型在专业领域的表现

## 快速开始

### 环境要求

- Python 3.10+
- OpenAI 兼容 API（OpenAI / DeepSeek / vLLM / Ollama 等）

### 安装

```bash
git clone https://github.com/hyl-ailab/dialogue-arena.git
cd dialogue-arena
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 API Key
```

### 运行

```bash
# 使用 config.yaml 中的默认问题
python main.py

# 自定义单个问题，3 轮辩论
python main.py --question "6G网络切片技术面临哪些核心挑战？" --rounds 3

# 指定并发数与详细日志
python main.py --concurrency 4 -v
```

### 输出目录

```
experiments/exp_20260625_143022/
├── experiment_meta.json              # 实验元信息
├── final_answers_finetune_multiturn.jsonl  # SFT 微调数据
├── arena_raw_log.jsonl               # 完整辩论日志
└── summary.json                      # 运行摘要
```

## 配置说明

编辑 [`config/config.yaml`](config/config.yaml)：

```yaml
llm:
  expert_model: gpt-4o-mini    # 专家模型
  judge_model: gpt-4o          # 裁判模型（建议更强）

arena:
  num_rounds: 5                # 每场辩论轮数
  max_concurrent_arenas: 8     # 并发场数

experts:                       # 自定义专家角色
  - id: itu_expert
    name: ITU-T 首席科学家
    persona: |
      你是...

seed_questions:                # 种子问题列表
  - "你的领域问题..."
```

### 本地模型（vLLM / Ollama）

```bash
# .env
OPENAI_API_KEY=sk-local
OPENAI_BASE_URL=http://localhost:8000/v1
```

## 项目结构

```
dialogue-arena/
├── arena/
│   ├── arena.py        # 单场辩论循环
│   ├── expert.py       # 专家并行回答
│   ├── judge.py        # 裁判评判（轮换+投票）
│   ├── runner.py       # 并发运行器
│   ├── llm_client.py   # OpenAI 兼容客户端
│   ├── prompts.py      # 提示词模板
│   └── utils.py        # 清洗、锁、实验目录
├── config/config.yaml
├── docs/architecture.svg
├── examples/
├── main.py
└── tests/
```

## 测试

```bash
pip install pytest
pytest tests/ -v
```

## 路线图

- [ ] 支持专家使用不同模型
- [ ] 模型自主提出初始问题
- [ ] Web 可视化辩论过程
- [ ] 导出 ShareGPT / Alpaca 等更多格式
- [ ] 集成 Embedding 去重与质量过滤

## 许可证

[MIT License](LICENSE)

## 引用

如果本项目对你的研究或工作有帮助，欢迎 Star 并引用：

```bibtex
@software{dialogue_arena2026,
  title  = {Dialogue Arena: Self-Evolving Multi-Model Dialogue Corpus Generation},
  author = {Dialogue Arena Contributors},
  year   = {2026},
  url    = {https://github.com/hyl-ailab/dialogue-arena}
}
```

## 结语

这套「自进化多模型多轮对话语料生成竞技场」通过模拟专家辩论、自动评判和迭代追问，实现了高质量对话数据的自动化生产。它不仅大幅降低了数据成本，更重要的是，生成的数据具备专业性、深度和逻辑性，是训练领域专家型 AI 模型的宝贵资产。

欢迎提交 Issue 和 PR，一起让 AI 自己「卷」起来！
