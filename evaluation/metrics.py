"""指标实现：确定性指标 + LLM 裁判。"""

from __future__ import annotations

import re

from rag.client import chat

# ══════════════════════════════════════════════════════
# 确定性指标（不花 token、可复现、无裁判偏差）
# ══════════════════════════════════════════════════════


def hit_at_k(hits: list[dict], expect_source: str | None) -> tuple[bool, int]:
    """期望来源是否出现在结果里？最高排第几？

    返回 (是否命中, 名次) 两个值。只看布尔命中会丢掉大量信息 ——
    第 1 名和第 4 名差别很大。

    ⚠️ 但这个指标在【候选同质】时会过度敏感：实测某题正确答案从第 3 名
       抖到第 5 名（未命中），可补上来的是同主题的另一篇文档，
       喂给生成器的信息量几乎没变，端到端指标也没恶化。
    """
    if expect_source is None:
        return (True, 0)  # 无标准答案的题不计入命中率
    for rank, h in enumerate(hits, 1):
        if h["source"] == expect_source:
            return (True, rank)
    return (False, 0)


def keyword_hit(answer: str, must_contain: list[str]) -> float:
    """答案里出现了几个必须词。返回命中率 0~1。"""
    if not must_contain:
        return 1.0
    a = answer.lower()
    return sum(1 for k in must_contain if k.lower() in a) / len(must_contain)


def honest_refusal(answer: str) -> bool:
    """对"库里没有"的题，是否老实说了不知道。"""
    a = answer.lower()
    return (
        ("不知道" in answer)
        or ("don't know" in a)
        or ("do not know" in a)
        or ("no information" in a)
        or ("not mentioned" in a)
    )


def mean_rank(ranks: list[int], penalty: int = 5) -> float:
    """平均名次。

    ⚠️ penalty 是【必须的】：未命中的题如果不计入，就会从分母里消失，
       制造幸存者偏差。实测某策略靠这招把"平均名次"从 2.14 刷到 1.20。
    """
    if not ranks:
        return 0.0
    return sum(r if r > 0 else penalty for r in ranks) / len(ranks)


def rank_spread(ranks: list[int], penalty: int = 5) -> float:
    """名次极差 —— 稳定性指标。

    同一题重跑多轮，名次的最大值和最小值之差。
    只测效果不测稳定性 = 没测：只在运气好时有效的方案不能上生产。
    """
    vals = [r if r > 0 else penalty for r in ranks]
    return float(max(vals) - min(vals)) if vals else 0.0


# ══════════════════════════════════════════════════════
# LLM 裁判（花 token，但能测"质量"）
# ══════════════════════════════════════════════════════

SCORE_PATTERN = re.compile(r"(?:score|分数)\s*[:：]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(raw: str, default: float = 5.0) -> float:
    """从裁判输出里抠分数。

    先找「分数：7」这种明确标记；找不到再退而求其次找任意数字。
    不直接用"第一个数字"—— 裁判可能在解释里先提到别的数字。
    """
    m = SCORE_PATTERN.search(raw)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return default
    return float(min(max(float(m.group(1)), 0.0), 10.0))


def judge_context_relevance(client, question: str, hits: list[dict]) -> float:
    """指标①：检索相关性 —— 召回的文档和问题相关吗？"""
    docs = "\n".join(f"- {h['text'][:400]}" for h in hits)
    prompt = (
        "你是严格的评估裁判。评估下面检索回来的文档与用户问题的相关性。\n"
        "能帮助回答问题给高分，无关给低分。\n"
        "最后只输出一行：分数：N（N 是 0 到 10 的整数）\n\n"
        f"【用户问题】{question}\n\n【检索到的文档】\n{docs}"
    )
    return parse_score(chat(client, prompt))


def judge_faithfulness(client, answer: str, hits: list[dict]) -> float:
    """指标②：忠实度 —— 答案有没有基于文档撒谎？"""
    docs = "\n".join(f"- {h['text'][:400]}" for h in hits)
    prompt = (
        "你是严格的评估裁判。判断答案是否完全基于提供的文档。\n"
        "- 答案每个说法都能在文档里找到依据 → 高分\n"
        "- 答案编造了文档里没有的内容（幻觉） → 低分\n"
        "- 答案说“我不知道”且文档确实没有相关信息 → 高分（诚实拒答是好的）\n"
        "最后只输出一行：分数：N（N 是 0 到 10 的整数）\n\n"
        f"【文档】\n{docs}\n\n【答案】{answer}"
    )
    return parse_score(chat(client, prompt))


def judge_answer_relevance(client, question: str, answer: str) -> float:
    """指标③：答案相关性 —— 答案有没有回答用户的问题？"""
    prompt = (
        "你是严格的评估裁判。判断答案是否真正回答了用户的问题。\n"
        "- 直接回答问题 → 高分\n"
        "- 答非所问、跑题 → 低分\n"
        "- 问题无法从材料回答、而答案诚实说“我不知道” → 7 分\n"
        "最后只输出一行：分数：N（N 是 0 到 10 的整数）\n\n"
        f"【用户问题】{question}\n\n【答案】{answer}"
    )
    return parse_score(chat(client, prompt))


def diagnose(ctx_rel: float, faith: float, ans_rel: float) -> str:
    bad = []
    if ctx_rel < 7:
        bad.append("检索相关性低→改检索")
    if faith < 7:
        bad.append("忠实度低→改生成防幻觉")
    if ans_rel < 7:
        bad.append("答案相关性低→改生成/query改写")
    return "；".join(bad) if bad else "良好"
