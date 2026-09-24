"""Query 改写：HyDE 与多查询展开。

════════════════════════════════════════════════════════════
⚠️ 重大警告：当语料比模型的知识新时，不要用 LLM 改写 query。
════════════════════════════════════════════════════════════
实测（Step 9）：本语料是新版 FastAPI 文档，安装写 `fastapi[standard]`；
而 glm-4-flash 的知识里还是【旧写法】`fastapi[all]`——该词在语料中出现 **0 次**。

HyDE 生成的"假设答案"把旧词写了进去，直接把检索带偏：
靶子题（安装问题）从"命中第 4 名"变成"完全未命中"。

同一个坑还出现在多查询展开的子问题里：
    `extras`（语料中 0 次）、`pagination`（语料中 0 次）。

**换句话说：LLM 会用它的旧词汇给你的检索投毒。**

所以本模块提供 find_oov_terms()：改写后先校验术语是否真的在语料里，
把"模型编的词"标出来。这一步比改写本身更重要。
"""

from __future__ import annotations

import re

from .client import chat

# ══════════════════════════════════════════════════════
# 语料术语校验
# ══════════════════════════════════════════════════════

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\[\]\.\-]*")


def build_vocab(texts: list[str]) -> set[str]:
    """从语料建词表，用于校验改写结果里有没有"编出来的词"。"""
    vocab: set[str] = set()
    for t in texts:
        vocab.update(m.lower() for m in TOKEN_PATTERN.findall(t))
    return vocab


def find_oov_terms(text: str, vocab: set[str]) -> list[str]:
    """找出改写文本里【语料中不存在】的术语。

    只检查"像 API 名/标识符"的词：含下划线、方括号、点，或者长度 >= 8。
    普通英文单词不查 —— 语料不可能覆盖所有介词和形容词。
    """
    oov: list[str] = []
    for m in TOKEN_PATTERN.findall(text):
        low = m.lower()
        if low in vocab:
            continue
        # 只关心"看起来像专有名词/API 名"的
        if any(ch in m for ch in "_[].") or (len(m) >= 8 and m[0].isupper()):
            oov.append(m)
    return sorted(set(oov))


def find_invented_terms(
    rewritten: str, user_context: str, vocab: set[str]
) -> list[str]:
    """找出【模型凭空加进来】的语料外术语。

    ⚠️ 为什么要和 find_oov_terms 分开：

        "语料外"和"模型编的"是两件事。用户自己提了一个语料里没有的术语时，
        改写把这个术语保留下来是【正确】的 —— 用户就是想问这个，
        回退到原问题反而把问题变得更难检索。

        只有当一个术语【只出现在改写结果里】、用户从头到尾没提过，
        才说明是模型在补内容时编进去的。

    实测区分这两种情况的例子：

        用户说 "那 WebSocketEndpoint 怎么用"
            改写 -> "How do I use WebSocketEndpoint in FastAPI?"
            语料外，但用户自己提的 -> **合法，不该回退**

        用户问 "那 CORS 呢"（历史里只有 CORS）
            改写 -> "Configure CORSMiddleware with allowed_origins"
            `CORSMiddleware` / `allowed_origins` 用户从没说过 -> **模型编的，回退**

    Args:
        rewritten: 改写后的查询
        user_context: 用户说过的话（当前追问 + 对话历史）
        vocab: 语料词表
    """
    context_lower = user_context.lower()
    return [
        term for term in find_oov_terms(rewritten, vocab)
        if term.lower() not in context_lower
    ]


# ══════════════════════════════════════════════════════
# HyDE
# ══════════════════════════════════════════════════════

HYDE_PROMPT = (
    "You are writing a hypothetical snippet from the FastAPI documentation.\n"
    "Write about 60 words of formal, statement-style technical documentation "
    "that would answer the question below. Write it in ENGLISH.\n"
    "It does NOT need to be factually correct - the FORM matters more than "
    "the facts. Use the vocabulary and API names the official docs would use.\n"
    "Output the documentation text only, no preamble.\n\n"
    "Question: {question}"
)


def hyde_query(client, question: str) -> str:
    """生成"假设答案"，用它（而非原问题）去检索。

    原理：embedding 模型是在文档语料上训练的，更熟悉陈述句。
    疑问句 -> 陈述句的"翻译"能让向量更近。

    ⚠️ 两重作用：① 形态对齐（问句->陈述句）② 词汇扩展（60 词里塞满术语）。
       实测中第 ② 重可能贡献更大 —— 一次纯关键词替换（无陈述句形态）
       反而比 HyDE 效果更好。但这是单次实验，不能下死结论。
    """
    return chat(client, HYDE_PROMPT.format(question=question)).strip()


# ══════════════════════════════════════════════════════
# 多查询展开
# ══════════════════════════════════════════════════════

MULTI_QUERY_PROMPT = (
    "Rewrite the user question below into {n} different, more specific search "
    "queries that would help find the answer in the FastAPI documentation.\n"
    "Vary the wording AND the angle: each query should approach the question "
    "from a different direction, not just rephrase it.\n"
    "Use the API names, keywords, and terms that the official docs would use.\n"
    "Output ONE query per line in ENGLISH, no numbering, no extra text.\n\n"
    "User question: {question}"
)


def multi_queries(client, question: str, n: int = 3) -> tuple[list[str], str | None]:
    """把一个问题拆成 n 个不同角度的子问题。

    返回 (子问题列表, 警告信息或 None)。

    ⚠️ Step 9 的两个教训都折进了这个函数：

    1. **必须检查条数。** 原版没做校验，LLM 只输出 1 行时程序照跑不误
       —— 静默降级，表面上一切正常，拿到的数据却和预想的不是一回事。

    2. **冗余 != 多样性。** 如果 3 个子问题只是同一句话的三种说法，
       RRF 融合就只是把同一个错误复制 3 份、还给它加权。
       实测：靶子题 3 个子问题全带 `dependencies`/`extras`/`packages`，
       全部撞同一个坑，多查询对该题【零改善】。
       所以 prompt 里显式要求 "Vary the wording AND the angle"。
    """
    raw = chat(client, MULTI_QUERY_PROMPT.format(n=n, question=question)).strip()

    lines: list[str] = []
    for line in raw.split("\n"):
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line:
            lines.append(line)

    warning = None
    if len(lines) < n:
        warning = f"只拆出 {len(lines)}/{n} 个子问题，本路结果不可比"

    return (lines[:n] or [question]), warning


# ══════════════════════════════════════════════════════
# RRF（改写后多路融合用）
# ══════════════════════════════════════════════════════


def merge_rewritten(results_list: list[list[dict]], k: int = 60, top_k: int = 4) -> list[dict]:
    """多查询各路的检索结果按名次融合。

    这里的 RRF 是【成立】的：几路都是同级的向量检索，只是 query 不同。
    （对比：把"向量召回"和"rerank"融合是不成立的 —— 那是上下游关系。）
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for results in results_list:
        for rank, hit in enumerate(results, 1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank)
            by_id[hit["id"]] = hit
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [dict(by_id[i], rrf=s) for i, s in ranked]
