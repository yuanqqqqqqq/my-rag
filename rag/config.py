"""全局配置：路径、模型、切块与检索参数。

实验脚本里常量是散落的（那是学习痕迹）；
一个真正的项目需要单点可改，所以全部收在这里。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# 语料：FastAPI 官方文档 tutorial/ 目录（51 篇 markdown）
#   若不存在，先跑 python scripts/fetch_data.py
DOCS_DIR = DATA_DIR / "fastapi-repo" / "docs" / "en" / "docs" / "tutorial"

# 代码示例：文档里的 {* ... docs_src/X.py *} 引用指令指向这里
#   不展开这些指令，文档里 271 处「怎么做」的答案就是个空壳 ——
#   实测：展开后某道题的答对率从 0/5 变成 5/5，在此之前三种检索方法都召不回正确答案。
DOCS_SRC_DIR = DATA_DIR / "docs_src"

CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "fastapi_docs"

# ── 模型 ──
EMBEDDING_MODEL = "embedding-3"
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "glm-4-flash")

# ── 切块 ──
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 1200      # 两段式切块里，第二段的递归切分上限
CHUNK_OVERLAP = 150
MIN_CHARS = 700        # 小于这个长度的块并入前一块

# ── 检索 ──
FINAL_K = 4            # 最终喂给生成器的块数
RECALL_K = 20          # 混合检索时每路的召回量
RERANK_POOL_K = 8      # rerank 的候选池大小
RRF_K = 60             # RRF 阻尼常数（越大越信"多路共识"）

# rerank 打分的并发度。
#
# 精排是【逐条打分】，每个候选一次 API 调用 —— 8 个候选串行要 3 秒左右，
# 而这 8 次调用彼此完全独立（不需要看见彼此的结果），所以可以并发。
# 实测 8 并发把它压到 0.4 秒左右，而且【结果和串行完全一致】。
#
# 设成 1 就是串行 —— 想复现"不加并发时有多慢"时可以这么用。
RERANK_WORKERS = int(os.getenv("RAG_RERANK_WORKERS", "8"))

# ── 智谱 embedding 的硬限制（实测，不是文档里的）──
#   单条 <= ~3072 tokens（本语料约 3.2 字符/token）；单次请求 <= 64 条。
#   超限报 1210「API 调用参数有误」—— 报错完全不提"太长"，很难查。
EMBED_BATCH_SIZE = 32
