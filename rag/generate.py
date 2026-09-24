"""答案生成：把检索到的材料拼进 prompt。

这一步刻意保持简单 —— 本项目的所有优化都在【检索侧】。
生成侧只做两件事：① 明确"只依据材料" ② 允许并鼓励说"我不知道"。
"""

from __future__ import annotations

from .client import chat, chat_stream

ANSWER_PROMPT = (
    "你是一个严谨的问答助手。请只根据下面提供的材料回答用户问题。"
    "如果材料里没有相关信息，请直接回答“我不知道”，不要编造。\n\n"
    "【材料】\n{context}\n\n【用户问题】{question}"
)

ANSWER_PROMPT_CITE = (
    "你是一个严谨的问答助手。请只根据下面提供的材料回答用户问题。"
    "如果材料里没有相关信息，请直接回答“我不知道”，不要编造。"
    "回答时请标注你依据的材料编号。\n\n"
    "【材料】\n{context}\n\n【用户问题】{question}"
)


def build_context(hits: list[dict]) -> str:
    """把命中的块拼成带来源标注的上下文。"""
    return "\n\n".join(
        f"【材料{i + 1}｜来自 {h['source']}】\n{h['text']}" for i, h in enumerate(hits)
    )


def generate_answer(
    client, question: str, hits: list[dict], *, cite: bool = False
) -> str:
    """基于检索到的材料生成答案。

    "我不知道" 是被允许的正确答案 —— 实测在库里没有的问题上，
    这套 prompt 能稳定拒答（队列 B 2/2）。
    """
    template = ANSWER_PROMPT_CITE if cite else ANSWER_PROMPT
    return chat(client, template.format(context=build_context(hits), question=question))


def generate_answer_stream(
    client, question: str, hits: list[dict], *, cite: bool = False
):
    """流式生成，逐块 yield 文本片段。

    和 generate_answer 共用同一套 prompt —— 流式只改变【怎么交付】，
    不改变【交付什么】。所以 `--stream` 不会影响答案质量。
    """
    template = ANSWER_PROMPT_CITE if cite else ANSWER_PROMPT
    prompt = template.format(context=build_context(hits), question=question)
    yield from chat_stream(client, prompt)
