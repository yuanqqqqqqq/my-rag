"""检索层：向量 / BM25 / 混合(RRF) / LLM rerank。

对外只暴露一个 `retrieve(strategy=...)`，CLI 和评估脚本共用。

四种策略，每一种的取舍都是实测出来的（不是拍脑袋）：
    vector   —— 基线，确定性最强。但在"词义歧义"的问题上会被带偏：
                实测它把「pip 安装依赖」和「依赖注入」搞混，正确答案排到第 4 名。
    hybrid   —— 向量 + BM25 用 RRF 融合。在含代码标识符的问题上有效，
                但在"词面陷阱"上会反向坑 —— BM25 分不清 parameter / parameters，
                中文里则是「申请单」和「申请」对不上。
    rerank   —— 默认方案：先扩召回，再用 LLM 精排。
                实测把靶子题的检索相关性从 3.0 提到 7.0。
    hybrid-rerank —— 两阶段：混合召回 + 精排。

⚠️ 关于 rerank 的一个坑（Step 11/12 实测）：
    LLM 打的是 0-10 的整数分。当候选同质时会出现大量平局
    （实测某题 8 个候选里 6 个都是 8 分），此时排序由随机的 1 分抖动决定。
    所以 rerank 的可靠性取决于【分差】，不取决于模型强弱。
    提高打分粒度（0-100）并不能解决 —— 拥挤只是换了刻度。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from rank_bm25 import BM25Okapi

from .client import chat, embed_one
from .config import FINAL_K, RECALL_K, RERANK_POOL_K, RERANK_WORKERS, RRF_K

# ══════════════════════════════════════════════════════
# 基础检索器：都返回 [{id, text, source, ...}]，按相关度降序
# ══════════════════════════════════════════════════════


def vector_hits(client, collection, query: str, top_k: int) -> list[dict]:
    """纯向量召回（bi-encoder）。"""
    res = collection.query(
        query_embeddings=[embed_one(client, query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": i, "text": t, "source": m["source"], "dist": d}
        for i, t, m, d in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]


# 分词实现移到了 rag/tokenize.py（中英文自动识别）。
# 这里重新导出，保持 `from rag.retrieve import tokenize` 这条路径可用。
from .tokenize import tokenize  # noqa: E402,F401


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def bm25_hits(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    """BM25 关键词召回（sparse）。"""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [dict(chunks[i], bm25_score=float(scores[i])) for i in order]


# ══════════════════════════════════════════════════════
# 融合
# ══════════════════════════════════════════════════════


def rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion：按【名次】融合多路结果。

    RRF(d) = Σ 1/(k + rank)

    为什么不把分数相加：向量给的是"距离"，BM25 给的是"词频分"，
    量纲不同没法加。RRF 只看名次，天然绕开这个问题；k 是阻尼，
    越大越信"多路共识"（k=60 时第 1 名和第 2 名几乎同权）。

    ⚠️ 适用条件：参与融合的几路必须是【同级的、独立的证据】。
       实测把"向量召回"和"rerank"按 RRF 融合会【变差】——
       因为它们是流水线的上下游，不是同级。rerank 的职责就是修正向量，
       让上游的错误排位拥有投票权，等于把错误平均进来（Step 11）。
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}

    for results in ranked_lists:
        for rank, hit in enumerate(results, 1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank)
            by_id[hit["id"]] = hit

    return [
        dict(by_id[i], rrf=s)
        for i, s in sorted(scores.items(), key=lambda x: -x[1])
    ]


# ══════════════════════════════════════════════════════
# LLM 精排
# ══════════════════════════════════════════════════════

SCORE_PATTERN = re.compile(r"(?:score|分数)\s*[:：]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

RERANK_PROMPT = (
    "You are a strict search relevance judge.\n"
    "Rate how well the document excerpt answers the user question.\n"
    "  10 = it directly and completely answers the question\n"
    "   0 = completely irrelevant\n"
    "Judge only whether the CONTENT answers the question, ignore writing style.\n"
    "Output ONLY one line: SCORE: N\n\n"
    "Question: {query}\n\n"
    "Document excerpt:\n{text}"
)


def parse_score(raw: str, default: float = 5.0) -> float:
    """从裁判输出里抠分数，钳到 0-10。

    先找 "SCORE: 7" 这种标记，找不到再退而求其次找任意数字 ——
    不直接用第一个数字，因为模型可能在解释里先提到别的数字。
    """
    m = SCORE_PATTERN.search(raw)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return default
    return float(min(max(float(m.group(1)), 0.0), 10.0))


def llm_rerank(client, query: str, hits: list[dict], top_k: int) -> list[dict]:
    """用 LLM 当 cross-encoder 精排。

    为什么能治"词面陷阱"：向量检索是 bi-encoder —— 问题和文档分别编码再比距离，
    只看得见"像不像"，看不见"答没答"。这里把问题和文档拼进同一个 prompt，
    模型能读懂"这段虽然在讲路径参数，但它讲的是特殊情况"。

    为什么【逐条打分】而不是"一次排 8 个"：
    逐条输出格式极稳定；一次排序在长列表上容易漏项/重项，解析失败就得降级。
    （曾经就因为解析没做校验，静默降级过。）

    为什么【并发】：
    这 N 次打分彼此完全独立 —— 不需要看见彼此的结果，只是最后一起排序。
    所以并发是零语义损失的优化：串行约 3 秒，8 并发约 0.4 秒，结果一模一样。
    """
    if not hits:
        return []

    if RERANK_WORKERS <= 1 or len(hits) == 1:
        scores = [_score_one(client, query, h) for h in hits]
    else:
        # map 保序，所以 scores[i] 一定对应 hits[i]
        with ThreadPoolExecutor(max_workers=min(RERANK_WORKERS, len(hits))) as pool:
            scores = list(pool.map(lambda h: _score_one(client, query, h), hits))

    scored = [dict(h, llm_score=s) for h, s in zip(hits, scores)]

    # 稳定排序：同分时保持【向量距离的顺序】，不引入新的不确定性。
    # （LLM 打分是 0-10 整数，候选同质时大量并列，稳定的 tie-break 很重要。）
    scored.sort(key=lambda x: x["llm_score"], reverse=True)
    return scored[:top_k]


def _score_one(client, query: str, hit: dict) -> float:
    raw = chat(client, RERANK_PROMPT.format(query=query, text=hit["text"][:600]))
    return parse_score(raw)


# ══════════════════════════════════════════════════════
# 统一入口
# ══════════════════════════════════════════════════════


def _recall_pool(client, collection, bm25, chunks, query: str, pool_k: int) -> list[dict]:
    """混合召回：两路各取 RECALL_K，RRF 融合，截到 pool_k。"""
    vec = vector_hits(client, collection, query, RECALL_K)
    bm = bm25_hits(bm25, chunks, query, RECALL_K)
    return rrf_fuse([vec, bm])[:pool_k]


def retrieve(
    client,
    collection,
    query: str,
    *,
    strategy: str = "rerank",
    top_k: int = FINAL_K,
    chunks: list[dict] | None = None,
    bm25=None,
) -> list[dict]:
    """统一检索入口。

    Args:
        strategy: vector | hybrid | rerank | hybrid-rerank
        chunks/bm25: hybrid 系策略需要；为 None 时按需从库加载。
    """
    needs_bm25 = strategy in ("hybrid", "hybrid-rerank")
    if needs_bm25 and (chunks is None or bm25 is None):
        from .store import load_chunks

        chunks = load_chunks(collection)
        bm25 = build_bm25(chunks)

    if strategy == "vector":
        return vector_hits(client, collection, query, top_k)

    if strategy == "hybrid":
        vec = vector_hits(client, collection, query, RECALL_K)
        bm = bm25_hits(bm25, chunks, query, RECALL_K)
        return rrf_fuse([vec, bm])[:top_k]

    if strategy == "rerank":
        pool = vector_hits(client, collection, query, RERANK_POOL_K)
        return llm_rerank(client, query, pool, top_k)

    if strategy == "hybrid-rerank":
        pool = _recall_pool(client, collection, bm25, chunks, query, RERANK_POOL_K)
        return llm_rerank(client, query, pool, top_k)

    raise ValueError(
        f"未知策略 {strategy!r}，可选：vector / hybrid / rerank / hybrid-rerank"
    )


STRATEGIES = ("vector", "hybrid", "rerank", "hybrid-rerank")
