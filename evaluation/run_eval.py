"""跑评估：对比多个检索策略。

    python -m rag.cli eval                              # vector vs rerank，各 1 轮
    python -m rag.cli eval --strategies vector rerank hybrid --runs 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.client import get_client
from rag.config import COLLECTION_NAME, FINAL_K
from rag.generate import generate_answer
from rag.retrieve import build_bm25, retrieve
from rag.store import get_collection, load_chunks

from .metrics import (
    diagnose,
    hit_at_k,
    honest_refusal,
    judge_answer_relevance,
    judge_context_relevance,
    judge_faithfulness,
    keyword_hit,
    mean_rank,
    rank_spread,
)

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
MISS_PENALTY = FINAL_K + 1  # 未命中按"比最后一名还差"计入


def load_eval_set(path: str | None = None) -> list[dict]:
    """读取评估集。默认用随项目提供的那份；传 path 则用自定义的。"""
    p = Path(path) if path else EVAL_SET_PATH
    if not p.exists():
        raise FileNotFoundError(f"评估集不存在：{p}")
    return json.loads(p.read_text(encoding="utf-8"))["cases"]


def run_strategy(
    client, collection, chunks, bm25, cases: list[dict], strategy: str, runs: int
) -> list[dict]:
    """跑一个策略 runs 轮，返回每个 case 的逐轮指标。"""
    print(f"\n{'=' * 78}")
    print(f"策略 {strategy}    （{len(cases)} 题 x {runs} 轮）")
    print("=" * 78)

    records: list[dict] = [[] for _ in cases]

    for run in range(1, runs + 1):
        if runs > 1:
            print(f"\n--- 第 {run}/{runs} 轮 ---")

        for ci, case in enumerate(cases):
            hits = retrieve(
                client, collection, case["q"],
                strategy=strategy, chunks=chunks, bm25=bm25,
            )
            answer = generate_answer(client, case["q"], hits)

            _, rank = hit_at_k(hits, case["expect_source"])
            rec = {
                "rank": rank,
                "kw": keyword_hit(answer, case["must_contain"]),
                "cr": judge_context_relevance(client, case["q"], hits),
                "fa": judge_faithfulness(client, answer, hits),
                "ar": judge_answer_relevance(client, case["q"], answer),
                "honest": honest_refusal(answer) if case["expect_source"] is None else None,
            }
            records[ci].append(rec)

            rank_s = f"第{rank}名" if rank else "未命中"
            print(
                f"  [{ci + 1}] {case['kind']:<8} {rank_s:<7} 关键词 {rec['kw']:.0%} | "
                f"相关性 {rec['cr']:>4.1f} 忠实度 {rec['fa']:>4.1f} 答案相关 {rec['ar']:>4.1f}"
            )

    return records


def summarize(records: list[dict], cases: list[dict], queue: str) -> dict:
    """汇总一个策略在某个队列上的指标。"""
    idx = [i for i, c in enumerate(cases) if c["queue"] == queue]
    per_case = [records[i] for i in idx]

    ranks = [r["rank"] for pc in per_case for r in pc]
    hit_ranks = [r for r in ranks if r > 0]

    out = {
        "n": sum(len(pc) for pc in per_case),
        "hits": len(hit_ranks),
        "mean_rank": mean_rank(ranks, MISS_PENALTY),
        "kw": _avg([r["kw"] for pc in per_case for r in pc]),
        "cr": _avg([r["cr"] for pc in per_case for r in pc]),
        "fa": _avg([r["fa"] for pc in per_case for r in pc]),
        "ar": _avg([r["ar"] for pc in per_case for r in pc]),
        "honest": sum(1 for pc in per_case for r in pc if r["honest"]),
        "spread": _avg([rank_spread([r["rank"] for r in pc], MISS_PENALTY) for pc in per_case]),
    }
    return out


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ══════════════════════════════════════════════════════


def main(
    strategies: list[str],
    runs: int = 1,
    index: str = COLLECTION_NAME,
    eval_set_path: str | None = None,
) -> int:
    cases = load_eval_set(eval_set_path)

    print("=" * 78)
    print("my-rag 评估")
    print("=" * 78)

    client = get_client()
    collection = get_collection(index)
    print(f"索引: {index}")
    print(f"库: {collection.count()} 块 | 题数: {len(cases)}"
          f"（队列 A {sum(1 for c in cases if c['queue'] == 'A')} / "
          f"队列 B {sum(1 for c in cases if c['queue'] == 'B')}）")

    chunks = load_chunks(collection)
    bm25 = build_bm25(chunks)

    all_records: dict[str, list[dict]] = {}
    for strategy in strategies:
        all_records[strategy] = run_strategy(
            client, collection, chunks, bm25, cases, strategy, runs
        )

    # ── 对比表 ──
    for queue, title in [
        ("A", "【队列 A】库里有答案的题  —— 这才是衡量 RAG 能力的"),
        ("B", "【队列 B】库里没有的题    —— 衡量它会不会瞎编"),
    ]:
        print(f"\n{'=' * 78}")
        print(title)
        print("=" * 78)
        print(f"  {'指标':<18}" + "".join(f"{s:>22}" for s in strategies))
        print("  " + "-" * (18 + 22 * len(strategies)))

        sums = {s: summarize(all_records[s], cases, queue) for s in strategies}

        def row(label: str, fn) -> None:
            print(f"  {label:<18}" + "".join(f"{fn(sums[s]):>22}" for s in strategies))

        if queue == "A":
            row("检索命中@K", lambda d: f"{d['hits']}/{d['n']}")
            row("平均命中名次", lambda d: f"{d['mean_rank']:.2f}")
            row("关键词命中率", lambda d: f"{d['kw']:.0%}")
            row("检索相关性", lambda d: f"{d['cr']:.1f}/10")
            row("忠实度", lambda d: f"{d['fa']:.1f}/10")
            row("答案相关性", lambda d: f"{d['ar']:.1f}/10")
            if runs > 1:
                row("名次波动", lambda d: f"{d['spread']:.2f}")

            print("\n  诊断（按 <7 分判断该改哪一层）：")
            for s in strategies:
                d = sums[s]
                print(f"    {s:<14} {diagnose(d['cr'], d['fa'], d['ar'])}")
        else:
            row("老实说不知道", lambda d: f"{d['honest']}/{d['n']}")
            row("忠实度", lambda d: f"{d['fa']:.1f}/10")

    print(f"\n  未命中的题按第 {MISS_PENALTY} 名计入平均名次（防幸存者偏差）。")
    if runs == 1:
        print("  ⚠️ 单轮结果含裁判噪声。要判断稳定性，加 --runs 3。")
    print("  ⚠️ 改进幅度 < 1 分不可信 —— LLM 裁判自己的飘动就有 ±1（实测）。")

    return 0


if __name__ == "__main__":
    sys.exit(main(strategies=["vector", "rerank"], runs=1))
