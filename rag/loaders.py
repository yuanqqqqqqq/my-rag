"""文档加载器：把各种格式的文件变成统一的文本表示。

支持 Markdown / 纯文本 / PDF / Word / 代码文件。

**每种格式返回的 kind 不同 —— 因为切块策略要跟着格式走：**

| kind | 切块策略 | 为什么 |
|---|---|---|
| markdown | 按标题层级切 | 保住文档结构，每块知道自己属于哪一节 |
| code | 按函数 / 类切 | 保住语义单元，别把函数劈成两半 |
| text / pdf / docx | 按段落递归切 | 没有结构信息可用，只能按长度兜底 |
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

# ══════════════════════════════════════════════════════
# 扩展名 → 类型
# ══════════════════════════════════════════════════════

MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
TEXT_EXTS = {".txt", ".rst", ".log", ".csv", ".tsv"}

CODE_EXTS = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".php": "php", ".swift": "swift",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "powershell",
    ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".json": "json", ".xml": "xml",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "css",
    ".vue": "vue", ".proto": "protobuf", ".tf": "hcl",
}

# 遍历目录时跳过的目录 —— 这些里面几乎不会有用户想检索的内容，
# 但会带来大量噪声和构建时间。
SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".env", "dist", "build", "target",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "chroma_db", ".next", ".cache", "vendor", "site-packages",
}

MAX_FILE_BYTES = 2 * 1024 * 1024  # 单文件上限 2M，跳过超大文件（如压缩日志）

# 永远不索引的文件名。
#
# 第一版没有这个名单，实测踩到两次：
#   1. domain.txt（分词词典）被当成正文索引进去了 —— 于是「年假 100 n」
#      这种词典行变成了一个检索结果
#   2. rag.toml 后缀是 .toml，正好在代码扩展名表里，
#      于是工具自己的配置文件成了语料
#
# 教训：**工具自己的配置/数据文件，必须由工具自己排除掉，
#        不能指望用户记得写 --exclude。**
SKIP_FILES = {".ragignore", "rag.toml", ".gitignore", ".gitattributes"}


@dataclass
class LoadedDoc:
    """一个加载好的文档。"""

    text: str
    source: str        # 相对路径，用于答案溯源
    kind: str          # markdown | text | pdf | docx | code
    lang: str = ""     # 代码文件的编程语言


# ══════════════════════════════════════════════════════
# 各格式的读取实现
# ══════════════════════════════════════════════════════


def _read_text(path: Path) -> str:
    """读文本文件。

    依次尝试 utf-8 / gbk —— 中文文档 GBK 编码很常见，
    直接 utf-8 读会抛异常，而异常信息对用户毫无帮助。
    """
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    """提取 PDF 文本。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "读 PDF 需要 pypdf：pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            # 标上页码 —— 溯源时"第几页"比"哪个文件"更有用
            pages.append(f"[第 {i} 页]\n{text}")
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    """提取 Word 文档的段落和表格。"""
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError(
            "读 Word 需要 python-docx：pip install python-docx"
        ) from exc

    document = docx.Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    # 表格单独提取 —— 很多文档的关键信息就在表里
    for ti, table in enumerate(document.tables, 1):
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"[表格 {ti}]\n" + "\n".join(rows))

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════
# 分发
# ══════════════════════════════════════════════════════


def kind_of(path: Path) -> tuple[str, str] | None:
    """判断文件类型。返回 (kind, lang)；不支持则返回 None。"""
    ext = path.suffix.lower()

    if ext in MARKDOWN_EXTS:
        return "markdown", ""
    if ext in TEXT_EXTS:
        return "text", ""
    if ext == ".pdf":
        return "pdf", ""
    if ext in (".docx", ".doc"):
        return "docx", ""
    if ext in CODE_EXTS:
        return "code", CODE_EXTS[ext]
    return None


def load_file(path: Path, root: Path) -> LoadedDoc | None:
    """加载单个文件。不支持或读取失败时返回 None（不中断整批）。"""
    detected = kind_of(path)
    if detected is None:
        return None

    kind, lang = detected
    try:
        if kind == "pdf":
            text = _load_pdf(path)
        elif kind == "docx":
            text = _load_docx(path)
        else:
            text = _read_text(path)
    except Exception as exc:  # noqa: BLE001 - 单个文件坏了不该中断整批
        print(f"  [跳过] {path.name}: {type(exc).__name__}: {exc}")
        return None

    if not text.strip():
        return None

    return LoadedDoc(
        text=text,
        source=os.path.relpath(path, root).replace(os.sep, "/"),
        kind=kind,
        lang=lang,
    )


def load_path(
    path: str | os.PathLike,
    *,
    exclude: list[str] | None = None,
    verbose: bool = True,
) -> list[LoadedDoc]:
    """加载一个文件或整个目录（递归）。

    自动跳过 .git / node_modules / __pycache__ 等目录 ——
    用户不会想检索这些，但它们的体积往往是内容的几十倍。

    Args:
        exclude: glob 模式列表，命中的文件不加载。
                 比如 ['*.txt', 'CHANGELOG*'] —— 把词典、日志这类
                 辅助文件挡在语料外面。
    """
    root_path = Path(path).resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"路径不存在：{root_path}")

    if root_path.is_file():
        root = root_path.parent
        doc = load_file(root_path, root)
        return [doc] if doc else []

    patterns = list(exclude or [])
    docs: list[LoadedDoc] = []
    skipped_big = skipped_dir = skipped_excl = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        # 原地修改 dirnames 才能真的剪掉遍历分支
        keep = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        skipped_dir += len(dirnames) - len(keep)
        dirnames[:] = keep

        for fn in sorted(filenames):
            if fn in SKIP_FILES:
                skipped_excl += 1
                continue
            if any(fnmatch.fnmatch(fn, pat) for pat in patterns):
                skipped_excl += 1
                continue

            fp = Path(dirpath) / fn
            try:
                if fp.stat().st_size > MAX_FILE_BYTES:
                    skipped_big += 1
                    continue
            except OSError:
                continue

            doc = load_file(fp, root_path)
            if doc:
                docs.append(doc)

    if verbose:
        kinds: dict[str, int] = {}
        for d in docs:
            kinds[d.kind] = kinds.get(d.kind, 0) + 1
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(kinds.items()))
        print(f"加载 {len(docs)} 个文件  ({summary})")
        if skipped_excl:
            print(f"  按 --exclude 跳过 {skipped_excl} 个文件")
        if skipped_big:
            print(f"  跳过 {skipped_big} 个超过 2M 的文件")
        if skipped_dir:
            print(f"  跳过 {skipped_dir} 个子目录（.git / node_modules 等）")

    return docs
