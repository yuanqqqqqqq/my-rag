"""多轮对话：让追问能引用前文。

核心难点只有一个 —— **追问本身不足以检索**：

    "那它怎么配置？"      ← 检索系统看不到"它"指什么
    "这个要多久？"        ← 同上

用户脑子里有上下文，向量检索没有。所以每轮都要先把追问改写成
一个**能独立成立的问句**，再去检索。

════════════════════════════════════════════════════════
⚠️ 但这个改写步骤本身就是 Step 9 那个「投毒」事故的同一个位置。
════════════════════════════════════════════════════════
实测过：让 LLM 改写 query，它会用自己的（可能过时的）知识往里塞词。
靶子题上 `fastapi[standard]` 被改写成了 `fastapi[all]` —— 后者在语料里出现 0 次。

所以这里的 prompt 加了三条例外规则，并且调用方可以用
`find_oov_terms()` 把改写结果里"语料中不存在的词"标出来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .client import chat

MAX_HISTORY_TURNS = 5

REWRITE_PROMPT = """你是一个查询改写器。根据对话历史，把用户的最新追问改写成
一个【能独立成立】的问题，让它在没有对话历史的情况下也能被理解。

规则（按重要性排序）：
1. 保持用户原本的措辞和语言，只补全指代（"它"、"这个"、"那个"指什么）。
2. **严禁引入用户没有提过的专有名词、API 名、版本号或术语。**
   如果不确定，宁可保留原样 —— 补错词比不补更糟。
3. 如果最新追问本身已经能独立成立，就原样输出，不要画蛇添足。
4. 只输出改写后的问题，不要解释、不要加引号。

对话历史：
{history}

最新追问：{question}

改写后的问题："""


def _format_history(turns: list[tuple[str, str]]) -> str:
    if not turns:
        return "(无)"
    lines = []
    for q, a in turns:
        lines.append(f"用户：{q}")
        lines.append(f"助手：{a[:120]}")
    return "\n".join(lines)


def rewrite_question(
    client, question: str, history: list[tuple[str, str]]
) -> tuple[str, str | None]:
    """把追问改写成独立问题。

    Returns:
        (改写后的问题, 警告信息或 None)

    没有历史时直接原样返回 —— 第一轮不需要改写，少一次 API 调用。
    """
    if not history:
        return question, None

    raw = chat(client, REWRITE_PROMPT.format(
        history=_format_history(history), question=question
    )).strip()

    # 模型有时会加引号或"改写后的问题是："这类前缀，剥掉
    cleaned = raw.strip().strip('"“”\'')
    cleaned = re.sub(r"^(改写后的问题|问题|Standalone question)\s*[:：]\s*", "",
                     cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.split("\n")[0].strip()

    # ⚠️ 静默降级检查（Step 9 的教训）：改写结果为空/过长/没变化 都要能接住
    if not cleaned:
        return question, "改写结果为空，已回落到原始追问"
    if len(cleaned) > len(question) * 6 + 100:
        return question, "改写结果异常冗长，已回落到原始追问"

    return cleaned, None


@dataclass
class Turn:
    question: str        # 用户原话
    standalone: str      # 改写后用于检索的问题
    answer: str
    hits: list[dict] = field(default_factory=list)
    rewritten: bool = False
    warning: str | None = None
    oov: list[str] = field(default_factory=list)   # 改写引入的"语料里没有的词"


class Conversation:
    """一场多轮对话。持有历史，逐轮检索 + 生成。"""

    def __init__(
        self,
        client,
        collection,
        *,
        name: str = "",
        strategy: str = "rerank",
        top_k: int = 4,
        chunks: list[dict] | None = None,
        bm25=None,
        vocab: set[str] | None = None,
        enable_rewrite: bool = True,
        guard_rewrite: bool = True,
        max_turns: int = MAX_HISTORY_TURNS,
    ):
        self.client = client
        self.collection = collection
        self.name = name
        self.strategy = strategy
        self.top_k = top_k
        self.chunks = chunks
        self.bm25 = bm25
        self.vocab = vocab
        self.enable_rewrite = enable_rewrite

        # 改写护栏：模型自己编了语料外的术语时，回退到原问题。
        # 关掉它就能观察"不设防时改写会跑到哪去" —— 对照实验用。
        self.guard_rewrite = guard_rewrite

        self.max_turns = max_turns
        self.turns: list[Turn] = []

    # ── 历史 ──
    def _history(self) -> list[tuple[str, str]]:
        return [(t.standalone, t.answer) for t in self.turns[-self.max_turns:]]

    def clear(self) -> None:
        self.turns.clear()

    # ── 一轮问答 ──
    def ask(self, question: str, *, stream: bool = False):
        """问一轮。stream=True 时逐块 yield 文本，并在结束后把 Turn 存在 self.last 上。"""
        from .generate import generate_answer, generate_answer_stream
        from .retrieve import retrieve
        from .rewrite import find_invented_terms

        standalone = question
        warning = None
        oov: list[str] = []

        if self.enable_rewrite:
            standalone, warning = rewrite_question(self.client, question, self._history())

            if self.vocab and standalone != question:
                # 只看【用户没说过】的语料外术语 —— 用户自己提的新术语是合法的，
                # 回退掉反而会让问题更难检索（详见 rewrite.find_invented_terms）
                user_context = question + " " + _format_history(self._history())
                oov = find_invented_terms(standalone, user_context, self.vocab)

                if oov and self.guard_rewrite:
                    # 模型自己编了语料里没有的术语 -> 这次改写不可信，退回原问题。
                    # 宁可少一次改写的收益，也不要让一个编出来的词把检索带偏。
                    standalone = question
                    warning = (
                        f"改写引入了语料外的词 {oov}，已回退到原问题"
                    )

        hits = retrieve(
            self.client, self.collection, standalone,
            strategy=self.strategy, top_k=self.top_k,
            chunks=self.chunks, bm25=self.bm25,
        )

        turn = Turn(
            question=question,
            standalone=standalone,
            answer="",
            hits=hits,
            rewritten=standalone != question,
            warning=warning,
            oov=oov,
        )

        if stream:
            pieces = []
            for piece in generate_answer_stream(self.client, standalone, hits):
                pieces.append(piece)
                yield piece
            turn.answer = "".join(pieces)
        else:
            turn.answer = generate_answer(self.client, standalone, hits)

        self.turns.append(turn)
        self.last = turn


def build_vocab_from_collection(collection) -> set[str]:
    """从索引内容建词表，用于检测改写引入的"编出来的词"。"""
    from .rewrite import build_vocab
    from .store import load_chunks

    return build_vocab([c["text"] for c in load_chunks(collection)])
