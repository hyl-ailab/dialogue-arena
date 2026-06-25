"""提示词模板。"""

from __future__ import annotations

SELF_CHECK_CRITERIA = """
在输出最终回答前，请在内心完成以下自我检查（不要将检查过程写入回答）：
1. 技术准确性：事实、标准编号、术语是否正确？
2. 完整性：是否覆盖了问题的核心要点？
3. 逻辑性：论述是否层层递进、无矛盾？
4. 专业深度：是否体现该领域的深度洞察，而非泛泛而谈？
5. 角色一致性：回答风格是否符合你的专家身份？
"""

EXPERT_ANSWER_PROMPT = """{persona}

{self_check}

当前讨论问题：
{question}

对话历史（如有）：
{history}

请以你的专家身份，针对上述问题给出专业、深入的回答。
要求：
- 直接输出最终回答，不要输出思考过程或标签
- 回答长度 300-600 字
- 可引用相关标准、协议或研究成果
"""

JUDGE_EVAL_PROMPT = """{persona}

你需要评估以下 {n} 个候选回答，针对问题：
「{question}」

{answers_block}

请严格按照 JSON 格式输出评判结果，不要包含其他文字：
{{
  "scores": [
    {{"candidate": "A", "accuracy": 0-10, "depth": 0-10, "logic": 0-10, "terminology": 0-10, "completeness": 0-10, "total": 0-50, "brief_comment": "一句话点评"}},
    ...
  ],
  "winner": "A"
}}

其中 winner 为总分最高的候选编号（A/B/C/D）。若并列，选择技术深度更突出者。
"""

FOLLOWUP_PROMPT = """你是一位资深技术主持人，正在主持一场专家圆桌讨论。

初始话题：{initial_question}

当前已进行到第 {round_num} 轮，本轮最佳回答来自 {winner_name}：
---
{winner_answer}
---

请基于上述最佳回答，提出一个更深入、更具挑战性的追问。
要求：
- 追问应推动讨论向技术细节或工程实践深入
- 长度 50-120 字
- 只输出追问本身，不要解释
"""

JUDGE_SELF_CHECK = """
在评判前，请客观公正，不受答案出现顺序影响。
只根据内容质量打分，不要偏袒任何特定风格。
"""


def format_history(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "（首轮，无历史）"
    lines = []
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"[{role}] {msg['content']}")
    return "\n".join(lines)


def format_answers_for_judge(answers: list[tuple[str, str, str]], labels: list[str]) -> str:
    """answers: [(expert_id, expert_name, content), ...]"""
    blocks = []
    for label, (_, name, content) in zip(labels, answers):
        blocks.append(f"--- 候选 {label}（{name}）---\n{content}")
    return "\n\n".join(blocks)
