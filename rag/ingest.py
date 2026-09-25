"""建索引管线：任意文档目录 -> 加载 -> 切块 -> 向量化 -> Chroma。

一条命令跑通：
    my-rag index ./my-docs --name mydocs

配置来源有三个，优先级从高到低：
    1. 命令行参数      最明确，覆盖一切
    2. rag.toml        语料目录里的索引参数
    3. .ragignore      语料目录里的排除规则

所以写好 rag.toml 之后，`my-rag index .` 就是完整的命令。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from .chunking import load_documents
from .client import get_client
from .config import CHUNK_SIZE, COLLECTION_NAME, DOCS_DIR, DOCS_SRC_DIR, MIN_CHARS
from .project import ProjectConfig, load_project_config
from .store import build_collection


def _resolve_config(
    paths: list[str],
    cli_name: str | None,
    cli_exclude: list[str] | None,
    cli_dict: str | None,
    cli_docs_src: str | None,
    verbose: bool,
) -> tuple[str, list[str], str | None, str | None]:
    """合并三个来源的配置。

    相对路径（词典、docs_src）按【语料目录】解析并转成绝对路径 ——
    这样从别的目录跑 `my-rag ask` 时，索引里存的路径依然有效。
    """
    project = ProjectConfig()
    base_dir: Path | None = None

    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            base_dir = pp.resolve()
            project = load_project_config(pp)
            break

    name = cli_name or project.name or COLLECTION_NAME

    exclude = list(cli_exclude or [])
    for pat in project.exclude:
        if pat not in exclude:
            exclude.append(pat)

    def _abs_from_cli(value: str | None) -> str | None:
        """命令行给的相对路径，按【当前目录】解析。

        用户在 shell 里敲 `--docs-src data/docs_src` 时，
        指的就是他现在所在目录下的那个 —— 不是语料目录下的。
        """
        if not value:
            return None
        return str(Path(value).resolve())

    def _abs_from_config(value: str | None) -> str | None:
        """rag.toml 里写的相对路径，按【语料目录】解析。

        配置是跟着语料走的：rag.toml 里写 `dict = "domain.txt"`
        意思是"和这份语料放在一起的 domain.txt"。
        """
        if not value:
            return None
        vp = Path(value)
        if vp.is_absolute() or base_dir is None:
            return str(vp)
        return str((base_dir / vp).resolve())

    dict_path = _abs_from_cli(cli_dict) or _abs_from_config(project.dict_path)
    docs_src = _abs_from_cli(cli_docs_src) or _abs_from_config(project.docs_src)

    if verbose and not project.is_empty():
        used = ["rag.toml" if (base_dir and (base_dir / "rag.toml").is_file()) else None,
                ".ragignore"]
        print(f"读到项目配置（{' + '.join(u for u in used if u)}）：")
        if project.name:
            print(f"  索引名  : {project.name}")
        if project.exclude:
            print(f"  排除规则: {', '.join(project.exclude)}")
        if project.dict_path:
            print(f"  分词词典: {project.dict_path}")
        if project.docs_src:
            print(f"  代码目录: {project.docs_src}")

    return name, exclude, dict_path, docs_src


def build_index(
    paths: list[str],
    name: str | None = None,
    *,
    docs_src_dir: str | None = None,
    exclude: list[str] | None = None,
    user_dict: str | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """从一批路径建索引。

    Args:
        paths: 文档目录或文件，可以给多个
        name: 索引名（不传则读 rag.toml，再退到默认名）
        docs_src_dir: FastAPI 文档专用。给了就展开 {* ... docs_src/X.py *} 引用指令
        exclude: glob 模式，命中的文件不进语料（会与 .ragignore 叠加）
        user_dict: 自定义分词词典（会与 rag.toml 合并）
        rebuild: 全量重建（改了切块参数时才需要）
        dry_run: 只加载和切块，不调 API、不写库

    Returns:
        {"added", "reused", "removed", "total", "mode", "chunks"} 统计信息
    """
    name, exclude, dict_path, docs_src = _resolve_config(
        paths, name, exclude, user_dict, docs_src_dir, verbose
    )

    # 词典要在切块/检索【之前】加载 —— 它影响 BM25 的分词
    if dict_path:
        from .tokenize import load_user_dict

        if not os.path.isfile(dict_path):
            raise FileNotFoundError(f"分词词典不存在：{dict_path}")
        n = load_user_dict(dict_path)
        if verbose:
            print(f"已加载分词词典：{n} 个词（{dict_path}）")

        # 词典文件本身不是语料，自动排除掉。
        # 实测踩过：domain.txt 被当成正文索引进去了，
        # 于是"年假 100 n"这种词典行变成了一个检索结果。
        dict_name = os.path.basename(dict_path)
        if dict_name not in exclude:
            exclude.append(dict_name)
            if verbose:
                print(f"（已自动排除词典文件 {dict_name}）")

    items: list[dict] = []
    for p in paths:
        if verbose:
            print(f"\n加载: {p}")
        part = load_documents(p, docs_src, exclude=exclude, verbose=verbose)
        items.extend(part)

    if not items:
        raise RuntimeError("没有切出任何块，检查路径是否正确")

    if verbose:
        _hint_split_terms(items)

    if dry_run:
        return _dry_run_report(items, name, verbose)

    client = get_client()
    stats = build_collection(
        client,
        items,
        name,
        rebuild=rebuild,
        config={
            "dict": dict_path,
            "chunk_size": CHUNK_SIZE,
            "min_chars": MIN_CHARS,
            "exclude": exclude,
            "docs_src": docs_src,
            "source_paths": [str(Path(p).resolve()) for p in paths],
            "built_at": datetime.now().isoformat(timespec="seconds"),
        },
        verbose=verbose,
    )
    stats["chunks"] = len(items)
    stats["name"] = name
    return stats


def _hint_split_terms(items: list[dict], top_n: int = 8) -> None:
    """提示可能被分词器切碎的领域词。

    中文没有空格，分词错了【不会报错】，只会让检索效果悄悄变差 ——
    用户会去怀疑 embedding、怀疑切块、怀疑模型，唯独想不到是分词的锅。
    这类问题必须由工具主动暴露出来，不能等用户自己发现。
    """
    from .tokenize import detect_split_terms, has_chinese

    texts = [it["text"] for it in items]
    if not any(has_chinese(t) for t in texts[:50]):
        return

    candidates = detect_split_terms(texts, top_n=top_n)
    if not candidates:
        return

    print("\n提示：以下词可能被分词器切碎了")
    for word, count in candidates:
        print(f"    {word}     (出现 {count} 次)")
    print("  中文没有空格，分词错了不会报错，只会让检索悄悄变差。")
    print("  如果它们在你的领域里是一个词，加进词典再重建：")
    print("    my-rag init .            # 生成词典模板")
    print("    my-rag index <路径> --dict domain.txt")


def _dry_run_report(items: list[dict], name: str, verbose: bool) -> dict:
    """预演：报告会索引什么，但不花一分钱 API 费用。

    为什么需要它：一次 `my-rag index .` 如果指错了目录
    （比如指到了整个用户主目录），会真的开始向量化几万个文件。
    先看一眼要花多少调用，是基本的安全网。
    """
    from .store import chunk_uid, get_collection

    kinds: dict[str, int] = {}
    sources: set[str] = set()
    for it in items:
        k = it.get("kind", "unknown")
        kinds[k] = kinds.get(k, 0) + 1
        sources.add(it["source"])

    uids = {chunk_uid(it["source"], it["text"]) for it in items}

    # 能对上现有索引的话，顺带算出增量会省多少
    added, reused = len(uids), 0
    legacy_note = ""
    try:
        existing = set(get_collection(name).get(include=[])["ids"])
        # 旧格式 id（chunk_0/chunk_1）来自更早的版本，对不上内容指纹，
        # 实际建库时会走全量重建 —— dry-run 的估算必须和真实行为一致，
        # 否则这个"安全网"会给出误导性的成本预估。
        if any(not i.startswith("c_") for i in existing):
            legacy_note = (
                f"检测到旧格式索引（{len(existing)} 个旧 id，来自更早的版本）。\n"
                f"  id 方案已升级为内容指纹，对不上 —— 本次会全量重建，无法复用。"
            )
        else:
            added = len(uids - existing)
            reused = len(uids & existing)
    except Exception:
        pass

    if verbose:
        print("\n" + "=" * 62)
        print("预演（dry-run）：不会调用任何 API")
        print("=" * 62)
        print(f"  索引名    : {name}")
        print(f"  来源文件  : {len(sources)} 个")
        print(f"  切块      : {len(items)} 块")
        print("  文档类型  :")
        for k, n in sorted(kinds.items()):
            print(f"    {k:<10} {n:>5} 块")
        if legacy_note:
            print(f"\n  ⚠️ {legacy_note}")
        print(f"\n  需要向量化: {added} 块")
        if reused:
            print(f"  可复用    : {reused} 块（内容没变，不花 API）")
        print(f"  预计调用  : 约 {-(-added // 32)} 次 embedding 请求")
        print("=" * 62)

    return {
        "added": added, "reused": reused, "removed": 0,
        "total": len(items), "mode": "dry-run", "chunks": len(items),
        "name": name,
    }


def build_knowledge_base(*, verbose: bool = True):
    """向后兼容：建内置的 FastAPI 示例索引。"""
    if verbose:
        print("使用内置 FastAPI 示例语料。")
        print(f"  文档: {DOCS_DIR}")
        print(f"  代码: {DOCS_SRC_DIR}")

    return build_index(
        [str(DOCS_DIR)], COLLECTION_NAME,
        docs_src_dir=str(DOCS_SRC_DIR), verbose=verbose,
    )
