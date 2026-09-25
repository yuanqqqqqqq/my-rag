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

# ── 简洁版 ──
#
# 生成耗时【完全由输出长度决定】，不是上下文长度：
# 实测同一份上下文，答案 294 字要 4.2 秒，1217 字要 20.8 秒。
# 而模型的默认风格很啰嗦 —— 一个"怎么上传文件"能写 1790 字。
#
# 加上下面这几句，实测耗时从 17.4s 降到 3.0s（约 6 倍），
# 关键词命中率和答案相关性都没有下降。
#
# ⚠️ 但它有代价，而且指标测不出来：
#     对"怎么用 X"这类问题，一句短答案够用；
#     对"A 和 B 有什么区别"这类需要列举的问题，短答案会把结构丢掉。
#     实测同一道比较题：详细版给了 6 点对比，简洁版压缩成了一句话。
#
# 所以这是个【用户选择】，不是可以优化掉的性能问题。默认用哪个见 cli.py。
BRIEF_SUFFIX = (
    "\n\nKeep the answer under 120 words. Be direct: give the answer, "
    "skip restating the materials and skip summaries. "
    "Always name the exact API/class."
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


def build_prompt(
    question: str, hits: list[dict], *, cite: bool = False, brief: bool = False
) -> str:
    template = ANSWER_PROMPT_CITE if cite else ANSWER_PROMPT
    prompt = template.format(context=build_context(hits), question=question)
    return prompt + BRIEF_SUFFIX if brief else prompt


def generate_answer(
    client, question: str, hits: list[dict], *, cite: bool = False, brief: bool = False
) -> str:
    """基于检索到的材料生成答案。

    "我不知道" 是被允许的正确答案 —— 实测在库里没有的问题上，
    这套 prompt 能稳定拒答。

    Args:
        brief: 简洁模式，约 6 倍快，但需要列举的问题会丢结构（见 BRIEF_SUFFIX）
    """
    return chat(client, build_prompt(question, hits, cite=cite, brief=brief))


def generate_answer_stream(
    client, question: str, hits: list[dict], *, cite: bool = False, brief: bool = False
):
    """流式生成，逐块 yield 文本片段。

    和 generate_answer 共用同一套 prompt —— 流式只改变【怎么交付】，
    不改变【交付什么】。所以 `--stream` 不会影响答案质量。
    """
    yield from chat_stream(client, build_prompt(question, hits, cite=cite, brief=brief))
