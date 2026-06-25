"""单元测试。"""

from arena.utils import build_finetune_record, clean_model_output


def test_clean_redacted_thinking():
    raw = "前缀<think>内部思考</think>后缀内容"
    assert clean_model_output(raw) == "前缀后缀内容"


def test_clean_think_tag():
    tag = "think"
    raw = f"可见<{tag}>内部思考</{tag}>不可见"
    assert clean_model_output(raw) == "可见不可见"


def test_clean_thinking_tag():
    raw = "A<thinking>long thought</thinking>B"
    assert clean_model_output(raw) == "AB"


def test_build_finetune_record():
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    record = build_finetune_record(messages)
    assert record == {"messages": messages}
