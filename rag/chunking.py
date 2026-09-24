"""切块：文档预处理 + 两段式切分。

纯函数，不碰网络，方便单独测试。

四步设计，每一步都对应一个实测出来的问题：
  1. 先展开 {* ... docs_src/X.py *} 引用指令 —— 不展开的话，
     文档里 271 处「怎么做」的答案全是空壳。
     实测：展开后某道题的答对率从 0/5 变成 5/5。
  2. 再按 Markdown 小标题切 —— 保住每块在文档里的位置。
  3. 对超长块递归切分 —— 一块混多个主题会稀释向量语义。
  4. 最后合并碎块 —— 几十字符的块语义太弱，会拉低召回质量。
"""

from __future__ import annotations

import os
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HEADERS_TO_SPLIT_ON,
    MIN_CHARS,
)

# 引用指令形如：{* ../../docs_src/path_params/tutorial001_py310.py hl[6:7] *}
# 捕获组是 docs_src/ 之后的相对路径
INCLUDE_PATTERN = re.compile(r"\{\*\s*[^*]*?docs_src/([^\s*]+)[^*]*\*\}")

# 代码块的语言标记，按文件扩展名给
EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".sh": "bash",
}

HEADER_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def resolve_includes(
    text: str, docs_src_dir: str
) -> tuple[str, int, int]:
    """把 {* ... docs_src/X.py ... *} 换成那个文件的真实代码。

    返回 (替换后的文本, 成功条数, 失败条数)。

    ⚠️ 找不到文件时【原样保留】（计入失败），不让整条管线崩掉 ——
       FastAPI 文档里确实有指向已删除文件的引用。
    """
    ok = 0
    fail = 0

    def _replace(match: re.Match) -> str:
        nonlocal ok, fail
        rel = match.group(1)  # 例如 path_params/tutorial001_py310.py
        path = os.path.join(docs_src_dir, rel)

        if not os.path.isfile(path):
            fail += 1
            return match.group(0)

        with open(path, encoding="utf-8") as f:
            code = f.read().rstrip()

        lang = EXT_LANG.get(os.path.splitext(path)[1].lower(), "")
        ok += 1
        return f"```{lang}\n{code}\n```"  # 包成代码块，保住 Markdown 结构

    return INCLUDE_PATTERN.sub(_replace, text), ok, fail


def chunk_document(text: str, source: str) -> list[dict]:
    """两段式切块：先按小标题切（保结构），再对超长块递归切（控大小）。"""
    items: list[dict] = []

    for hc in HEADER_SPLITTER.split_text(text):
        body = hc.page_content
        pieces = (
            [body] if len(body) <= CHUNK_SIZE else RECURSIVE_SPLITTER.split_text(body)
        )
        for piece in pieces:
            items.append(
                {
                    "text": piece,
                    "source": source,
                    "h1": hc.metadata.get("h1", ""),
                    "h2": hc.metadata.get("h2", ""),
                    "h3": hc.metadata.get("h3", ""),
                }
            )

    return items


def merge_tiny(items: list[dict], min_chars: int = MIN_CHARS) -> list[dict]:
    """把过小的块并入前一块（合并后不超过 CHUNK_SIZE）。

    为什么：几十字符的碎块向量语义很弱，会拉低召回质量。
    ⚠️ 这个阈值不能随便调 —— 实测 MIN_CHARS=700 时某道题的答对率会从 2/5 掉到 0/5。
       改之前先跑 eval/run_eval.py 对比。
    """
    merged: list[dict] = []
    for it in items:
        if (
            merged
            and len(it["text"]) < min_chars
            and len(merged[-1]["text"]) + len(it["text"]) + 2 <= CHUNK_SIZE
        ):
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + it["text"]
        else:
            merged.append(dict(it))
    return merged


def chunk_text(text: str, source: str) -> list[dict]:
    """纯文本切块（txt / pdf / docx）。

    这些格式没有可用的结构信息，只能按段落递归切。
    """
    return [
        {"text": p, "source": source, "h1": "", "h2": "", "h3": ""}
        for p in RECURSIVE_SPLITTER.split_text(text)
        if p.strip()
    ]


# 代码切块的分隔符：优先在函数/类边界断开，别把函数劈成两半
CODE_SEPARATORS: dict[str, list[str]] = {
    "python": ["\nclass ", "\ndef ", "\nasync def ", "\n\n\n", "\n\n", "\n", " "],
    "javascript": ["\nfunction ", "\nclass ", "\nexport ", "\nconst ", "\n\n", "\n", " "],
    "typescript": ["\nfunction ", "\nclass ", "\nexport ", "\ninterface ", "\n\n", "\n", " "],
    "java": ["\npublic ", "\nprivate ", "\nclass ", "\n\n", "\n", " "],
    "go": ["\nfunc ", "\ntype ", "\n\n", "\n", " "],
    "rust": ["\nfn ", "\nimpl ", "\nstruct ", "\n\n", "\n", " "],
}
DEFAULT_CODE_SEPARATORS = ["\nclass ", "\nfunction ", "\ndef ", "\n\n", "\n", " "]


def chunk_code(text: str, source: str, lang: str = "") -> list[dict]:
    """按函数/类边界切代码。

    代码和散文不一样：在函数中间切开，切出来的块既读不懂也检不到。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CODE_SEPARATORS.get(lang, DEFAULT_CODE_SEPARATORS),
    )
    return [
        {"text": p, "source": source, "h1": "", "h2": "", "h3": ""}
        for p in splitter.split_text(text)
        if p.strip()
    ]


def chunk_loaded(doc, *, min_chars: int = MIN_CHARS) -> list[dict]:
    """按文档类型分发切块策略。doc 是 rag.loaders.LoadedDoc。"""
    if doc.kind == "markdown":
        items = chunk_document(doc.text, doc.source)
    elif doc.kind == "code":
        items = chunk_code(doc.text, doc.source, doc.lang)
    else:
        items = chunk_text(doc.text, doc.source)

    for it in items:
        it["kind"] = doc.kind
        it["lang"] = doc.lang

    return merge_tiny(items, min_chars)


def load_documents(
    docs_dir: str,
    docs_src_dir: str | None = None,
    *,
    exclude: list[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """读取目录 -> 加载 -> （可选）展开引用指令 -> 切块 -> 合并碎块。

    Args:
        docs_dir: 语料根目录，支持 md / txt / pdf / docx / 代码文件
        docs_src_dir:
            FastAPI 文档专用。给了就展开 {* ... docs_src/X.py *} 引用指令；
            不给就跳过 —— 这是可选的预处理，不是所有语料都有这种引用。
        exclude: glob 模式，命中的文件不加载

    返回带 source / 标题层级 / chunk_index 的块列表。
    """
    from .loaders import load_path  # 延迟导入，避免与 loaders 形成环

    docs = load_path(docs_dir, exclude=exclude, verbose=verbose)
    if not docs:
        raise FileNotFoundError(
            f"在 {docs_dir} 里没找到任何可索引的文件。\n"
            f"  支持：Markdown / txt / PDF / Word / 常见代码文件"
        )

    items: list[dict] = []
    total_ok = total_fail = 0
    chars_before = chars_after = 0

    for doc in docs:
        if docs_src_dir and doc.kind == "markdown":
            chars_before += len(doc.text)
            doc.text, ok, fail = resolve_includes(doc.text, docs_src_dir)
            chars_after += len(doc.text)
            total_ok += ok
            total_fail += fail

        items.extend(chunk_loaded(doc))

    if verbose and docs_src_dir:
        print(f"引用指令: 成功展开 {total_ok} 条, 失败 {total_fail} 条")
        print(f"语料字符: {chars_before} -> {chars_after} (+{chars_after - chars_before})")
    if verbose:
        print(f"切块合并后: {len(items)} 块")

    for i, it in enumerate(items):
        it["chunk_index"] = i

    return items
