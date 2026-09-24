"""智谱 API 封装。

所有网络调用都收在这里，方便统一加超时、重试、限流和用量统计。
其余模块只依赖本文件的三个函数。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

from .config import CHAT_MODEL, EMBED_BATCH_SIZE, EMBEDDING_MODEL, PROJECT_ROOT


def get_client() -> ZhipuAI:
    """读取 .env 并建客户端。

    显式传 .env 路径，不依赖 load_dotenv() 的默认查找行为 ——
    这样无论从哪个目录、哪个入口调用，读到的都是同一个文件。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("xxxx"):
        raise RuntimeError(
            f"没读到 ZHIPUAI_API_KEY。\n"
            f"  请确认 {PROJECT_ROOT / '.env'} 存在，内容形如：\n"
            f"    ZHIPUAI_API_KEY=你的key\n"
            f"  可以从 {PROJECT_ROOT / '.env.example'} 复制一份。"
        )
    return ZhipuAI(api_key=api_key)


def chat(client: ZhipuAI, prompt: str, model: str | None = None) -> str:
    """单轮对话。RAG 里的生成、改写、裁判全走这里。"""
    resp = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def chat_stream(client: ZhipuAI, prompt: str, model: str | None = None):
    """流式对话：逐块 yield 文本片段。

    为什么要流式：默认策略每次查询要 8 次精排调用（约 3 秒）再生成（约 3 秒），
    用户盯着空屏幕等 6 秒。流式把「首字延迟」降到 1 秒内 ——
    总时长没变，但感知完全不同。

    Args:
        prompt: 完整提示词。RAG 的流式只用在【生成】这一步 ——
                精排必须拿到完整分数才能排序，没法流式。
    """
    stream = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        # 有些 chunk 只带 usage 或 role，choices 可能为空
        if not getattr(chunk, "choices", None):
            continue
        piece = getattr(chunk.choices[0].delta, "content", None)
        if piece:
            yield piece


def embed(client: ZhipuAI, texts: list[str]) -> list[list[float]]:
    """批量向量化，自动按 EMBED_BATCH_SIZE 分批。

    返回顺序严格对应输入顺序 —— 按 index 排序，不依赖 API 的返回次序。
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(
            item.embedding for item in sorted(resp.data, key=lambda x: x.index)
        )
    return vectors


def embed_one(client: ZhipuAI, text: str) -> list[float]:
    """单条向量化（查询侧用）。"""
    return embed(client, [text])[0]
