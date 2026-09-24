"""命令行入口。

    my-rag index ./my-docs --name mydocs     建索引（md/txt/pdf/docx/代码）
    my-rag ask "问题"                         提问
    my-rag ask "问题" --index mydocs          指定索引
    my-rag retrieve "问题"                    只看检索结果
    my-rag trace "问题"                       逐层看检索管线（调检索用）
    my-rag list                               列出所有索引
    my-rag info                               索引详情
    my-rag drop mydocs                        删除索引
    my-rag eval                               跑评估

也可以 `python -m rag.cli ...` 或 `python -m rag ...`
"""

from __future__ import annotations

import argparse
import sys
import time

from .config import COLLECTION_NAME

STRATEGY_CHOICES = ["vector", "hybrid", "rerank", "hybrid-rerank"]


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，打印中文会撞编码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _open(args: argparse.Namespace, strategy: str | None = None):
    """打开索引 + 建好该策略需要的本地索引。返回 (client, collection, chunks, bm25)。

    **会自动应用建索引时记录的分词词典。**
    这一步很关键：如果建索引时用了词典、查询时忘了带，
    查询和文档的切法就对不上，BM25 会【静默失效】——
    不报错，只是检索质量悄悄变差，很难发现。
    """
    import os

    from .client import get_client
    from .retrieve import build_bm25
    from .store import get_collection, get_index_config, load_chunks

    strategy = strategy or args.strategy

    # 命令行显式给的词典优先；没给就回落到索引里存的
    if not getattr(args, "user_dict", None):
        saved = get_index_config(args.index).get("dict")
        if saved and os.path.isfile(saved):
            from .tokenize import load_user_dict

            load_user_dict(saved)
            if getattr(args, "verbose_config", False):
                print(f"（已自动应用索引里记录的分词词典：{saved}）")

    client = get_client()
    collection = get_collection(args.index)

    chunks = bm25 = None
    if strategy in ("hybrid", "hybrid-rerank"):
        chunks = load_chunks(collection)
        bm25 = build_bm25(chunks)

    return client, collection, chunks, bm25


# ══════════════════════════════════════════════════════
# 索引管理
# ══════════════════════════════════════════════════════


def cmd_index(args: argparse.Namespace) -> int:
    from .ingest import build_index

    import json

    # 词典交给 build_index 统一处理 —— 它还要把词典路径写进索引，
    # 并且相对路径要按语料目录解析。这里再加载一次会重复。
    stats = build_index(
        args.paths,
        args.name,
        docs_src_dir=args.docs_src,
        exclude=args.exclude,
        user_dict=args.user_dict,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif not args.dry_run:
        # 用解析后的实际索引名 —— 它可能来自 rag.toml，args.name 会是 None
        print(f'\n下一步：my-rag ask "你的问题" --index {stats.get("name", args.name)}')

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from .store import list_collections

    items = list_collections()
    if not items:
        print("还没有任何索引。")
        print("  建一个：my-rag index ./你的文档目录 --name mydocs")
        return 0

    print(f"{'索引名':<28} {'块数':>8}")
    print("-" * 38)
    for it in items:
        print(f"{it['name']:<28} {it['count']:>8}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .store import collection_info, get_index_config

    info = collection_info(args.index)
    print(f"索引: {info['name']}")
    print(f"  块数      : {info['chunks']}")
    print(f"  来源文件数: {info['sources']}")
    print("  文档类型  :")
    for kind, n in sorted(info["kinds"].items()):
        print(f"    {kind:<12} {n:>5} 块")

    cfg = get_index_config(args.index)
    if cfg:
        print("\n  建索引时的参数（ask 时会自动应用）：")
        for key in ("dict", "chunk_size", "min_chars", "exclude",
                    "docs_src", "built_at", "source_paths"):
            value = cfg.get(key)
            if value:
                shown = ", ".join(map(str, value)) if isinstance(value, list) else value
                print(f"    {key:<12} {shown}")
    return 0


RAGIGNORE_TEMPLATE = """\
# my-rag 排除规则（语法同 .gitignore，一行一个 glob）
# 命中的文件不会进入语料。

# 日志与临时文件
*.log
*.tmp
*.bak

# 草稿
draft_*
*_draft.*

# 数据文件
*.csv
*.json
"""

RAGTOML_TEMPLATE = """\
# my-rag 的项目配置。
# 放在语料目录里，`my-rag index .` 就会自动读取。

# 索引名（不写则用默认名 fastapi_docs）
# name = "my-docs"

# 分词词典。中文语料强烈建议配一个 —— jieba 的通用词典不认识领域词，
# 会把「年假」切成「年」「假」。
# 格式：每行一个词，可选跟词频和词性，例如：  年假 100 n
# dict = "domain.txt"

# 额外排除的文件（会和 .ragignore 叠加）
# exclude = ["*.min.js", "vendor/*"]

# FastAPI 官方文档专用：展开 Markdown 里的 {* ... docs_src/X.py *} 引用指令
# docs_src = "../docs_src"
"""


def cmd_init(args: argparse.Namespace) -> int:
    """在语料目录里生成 .ragignore 和 rag.toml 模板。"""
    from pathlib import Path

    target = Path(args.path).resolve()
    if not target.is_dir():
        print(f"[错误] 不是目录：{target}", file=sys.stderr)
        return 1

    created = []
    for filename, body in (
        (".ragignore", RAGIGNORE_TEMPLATE),
        ("rag.toml", RAGTOML_TEMPLATE),
    ):
        path = target / filename
        if path.exists() and not args.force:
            print(f"  [跳过] {filename} 已存在（--force 可覆盖）")
            continue
        path.write_text(body, encoding="utf-8")
        created.append(filename)

    if created:
        print(f"已生成：{', '.join(created)}")
        print("\n按需修改后，直接跑：")
        print(f'  my-rag index "{target}"')
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    from .store import drop_collection

    if not args.yes:
        answer = input(f"确认删除索引 {args.name!r}？(y/N) ").strip().lower()
        if answer != "y":
            print("已取消。")
            return 0

    if drop_collection(args.name):
        print(f"已删除索引 {args.name!r}")
        return 0

    print(f"索引 {args.name!r} 不存在。")
    return 1


# ══════════════════════════════════════════════════════
# 检索与问答
# ══════════════════════════════════════════════════════


def _hits_to_json(hits: list[dict]) -> list[dict]:
    """把命中结果转成可序列化的结构（去掉内部字段，分数保留合理精度）。"""
    out = []
    for i, h in enumerate(hits, 1):
        item = {"rank": i, "source": h["source"], "text": h["text"]}
        if "dist" in h:
            item["distance"] = round(float(h["dist"]), 4)
        if "llm_score" in h:
            item["llm_score"] = float(h["llm_score"])
        if "bm25_score" in h:
            item["bm25_score"] = round(float(h["bm25_score"]), 4)
        if "rrf" in h:
            item["rrf"] = round(float(h["rrf"]), 6)
        out.append(item)
    return out


def _retrieve_step(client, collection, chunks, bm25, question: str, args):
    from .retrieve import retrieve

    t0 = time.perf_counter()
    hits = retrieve(
        client, collection, question,
        strategy=args.strategy, top_k=args.top_k, chunks=chunks, bm25=bm25,
    )
    return hits, time.perf_counter() - t0


def _generate_step(client, question: str, hits: list[dict], *, stream: bool):
    """生成答案。stream=True 时边收边打，同时把完整文本拼好返回。

    注意：流式输出【必须】在打印完标题和来源之后调用 ——
    否则答案会先于"问题：xxx"出现在屏幕上，顺序就乱了。
    所以检索和生成拆成了两步，由调用方编排顺序。
    """
    from .generate import generate_answer, generate_answer_stream

    t0 = time.perf_counter()
    if stream:
        pieces = []
        for piece in generate_answer_stream(client, question, hits):
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()   # 不 flush 会被缓冲住，看起来还是"卡住"
        answer = "".join(pieces)
        sys.stdout.write("\n")
    else:
        answer = generate_answer(client, question, hits)
    return answer, time.perf_counter() - t0


def _ask_one(client, collection, chunks, bm25, question: str, args, *, header: bool):
    """跑一个问题。返回结果 dict。

    打印顺序刻意设计成：标题 → 材料来源 → 答案 → 耗时。
    把耗时放最后，是因为流式模式下它只有在答案收完之后才知道。
    """
    show = not args.json
    stream = args.stream and not args.json

    if header and show:
        print(f"\n{'─' * 70}")
        print(f"问题: {question}")
        print(f"索引: {args.index}   策略: {args.strategy}")
        print(f"{'─' * 70}")

    hits, t_ret = _retrieve_step(client, collection, chunks, bm25, question, args)

    if show and args.show_hits:
        print("\n检索到的材料：")
        for i, h in enumerate(hits, 1):
            extra = f"  llm={h['llm_score']:.0f}" if "llm_score" in h else ""
            print(f"  [{i}] {h['source']}{extra}")
        print()

    answer, t_gen = _generate_step(client, question, hits, stream=stream)

    if show:
        if not stream:
            print(answer)
        print(f"\n[耗时] 检索 {t_ret:.2f}s + 生成 {t_gen:.2f}s")

    return {
        "question": question,
        "index": args.index,
        "strategy": args.strategy,
        "elapsed_seconds": {"retrieval": round(t_ret, 3), "generation": round(t_gen, 3)},
        "hits": _hits_to_json(hits),
        "answer": answer,
    }


def cmd_ask(args: argparse.Namespace) -> int:
    import json

    _apply_user_dict(args)
    client, collection, chunks, bm25 = _open(args)

    questions = [args.question] if args.question else []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            questions = [
                ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")
            ]
    if not questions:
        print("[错误] 没有要问的问题（给一个 question，或用 --file）。", file=sys.stderr)
        return 1

    batch = args.file is not None
    results = []

    for qi, question in enumerate(questions, 1):
        if batch and not args.json:
            print(f"\n{'━' * 70}")
            print(f"[{qi}/{len(questions)}]")
            print(f"{'━' * 70}")

        results.append(
            _ask_one(client, collection, chunks, bm25, question, args, header=True)
        )

    if args.json:
        # 批量用 JSONL（一行一条），便于管道里逐条消费；单条用常规 JSON
        if batch:
            for r in results:
                print(json.dumps(r, ensure_ascii=False))
        else:
            print(json.dumps(results[0], ensure_ascii=False, indent=2))

    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    from .retrieve import retrieve

    _apply_user_dict(args)
    client, collection, chunks, bm25 = _open(args)

    t0 = time.perf_counter()
    hits = retrieve(
        client, collection, args.question,
        strategy=args.strategy, top_k=args.top_k, chunks=chunks, bm25=bm25,
    )
    elapsed = time.perf_counter() - t0

    if args.json:
        import json

        print(json.dumps({
            "question": args.question,
            "index": args.index,
            "strategy": args.strategy,
            "elapsed_seconds": round(elapsed, 3),
            "hits": _hits_to_json(hits),
        }, ensure_ascii=False, indent=2))
        return 0

    print(f"\n问题: {args.question}")
    print(f"索引: {args.index}   策略: {args.strategy}   {len(hits)} 条   {elapsed:.2f}s\n")

    for i, h in enumerate(hits, 1):
        bits = []
        if "dist" in h:
            bits.append(f"dist={h['dist']:.4f}")
        if "llm_score" in h:
            bits.append(f"llm={h['llm_score']:.0f}")
        if "rrf" in h:
            bits.append(f"rrf={h['rrf']:.5f}")
        print(f"  [{i}] {h['source']:<42} {' '.join(bits)}")
        print(f"      {h['text'][:88].replace(chr(10), ' ')}...")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    """逐层打印检索管线 —— 看清每一步数据变成了什么样。

    既是教学工具，也是调检索时的排查器：
    结果不对时，你能一眼看出是「召回没召到」还是「排序排错了」。
    """
    from collections import Counter

    from .client import chat
    from .config import FINAL_K, RECALL_K, RERANK_POOL_K, RRF_K
    from .generate import generate_answer
    from .retrieve import (
        RERANK_PROMPT,
        bm25_hits,
        build_bm25,
        parse_score,
        rrf_fuse,
        vector_hits,
    )
    from .store import load_chunks

    _apply_user_dict(args)
    client, collection, _, _ = _open(args, "vector")   # trace 自己建 bm25，这里不重复建
    q = args.question
    strategy = args.strategy

    uses_bm25 = strategy in ("hybrid", "hybrid-rerank")
    uses_rerank = strategy in ("rerank", "hybrid-rerank")

    print("=" * 78)
    print(f"问题: {q}")
    print(f"索引: {args.index}   策略: {strategy}")
    print("=" * 78)

    # ── 【1】向量召回 ──
    vec = vector_hits(client, collection, q, RECALL_K)
    show = RERANK_POOL_K if uses_rerank else FINAL_K
    print(f"\n【1】向量召回 top-{RECALL_K}   bi-encoder：问题和文档分别编码，比余弦距离")
    for i, h in enumerate(vec, 1):
        mark = "   <- 后续只保留到这里" if uses_rerank and i == show else ""
        print(f"   {i:>2}. [{h['dist']:.4f}] {h['source']}{mark}")

    pool = vec[:show]

    # ── 【2】BM25 + 【3】RRF 融合 ──
    if uses_bm25:
        chunks = load_chunks(collection)
        bm25 = build_bm25(chunks)
        bm = bm25_hits(bm25, chunks, q, RECALL_K)
        print(f"\n【2】BM25 关键词召回 top-{RECALL_K}   sparse：词频分，看得见字面看不见语义")
        for i, h in enumerate(bm, 1):
            print(f"   {i:>2}. [{h['bm25_score']:>8.2f}] {h['source']}")

        fused = rrf_fuse([vec, bm])[:show]
        print(f"\n【3】RRF 融合 top-{show}   k={RRF_K}：只用名次，不用分数（量纲不同没法加）")
        for i, h in enumerate(fused, 1):
            print(f"   {i:>2}. [{h['rrf']:.5f}] {h['source']}")
        pool = fused

    # ── 【4】LLM 精排 ──
    if uses_rerank:
        print(f"\n【4】LLM 精排：逐条打分（{len(pool)} 次 API 调用）")
        scored = []
        for h in pool:
            raw = chat(client, RERANK_PROMPT.format(query=q, text=h["text"][:600]))
            s = parse_score(raw)
            scored.append(dict(h, llm_score=s))
            print(f"   llm={s:>4.0f}   {h['source']}")
        scored.sort(key=lambda x: x["llm_score"], reverse=True)
        pool = scored

        ties = Counter(h["llm_score"] for h in pool[:FINAL_K])
        print(
            f"\n   可靠性判据：最终 top-{FINAL_K} 里最大并列组 = {max(ties.values())} 个"
        )
        print("      并列越大 → LLM 越分不出高下 → 排序越靠运气（改粒度也没用，实测）")

    # ── 【5】最终结果 ──
    final = pool[:FINAL_K]
    print(f"\n【5】最终喂给生成器的 top-{FINAL_K}")
    for i, h in enumerate(final, 1):
        bits = []
        if "dist" in h:
            bits.append(f"dist={h['dist']:.4f}")
        if "llm_score" in h:
            bits.append(f"llm={h['llm_score']:.0f}")
        if "rrf" in h:
            bits.append(f"rrf={h['rrf']:.5f}")
        print(f"   [{i}] {h['source']:<44} {' '.join(bits)}")

    # ── 【6】生成 ──
    if args.answer:
        print("\n【6】生成答案")
        print(generate_answer(client, q, final))
    else:
        print("\n（加 --answer 可以顺带生成答案）")

    return 0


# ══════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════


def cmd_chat(args: argparse.Namespace) -> int:
    """交互式多轮对话。

    追问本身不足以检索（"那它怎么配置？"），所以每轮先把追问改写成
    能独立成立的问题。用 --no-rewrite 可以关掉，用来做对照。
    """
    from .conversation import Conversation, build_vocab_from_collection

    _apply_user_dict(args)
    client, collection, chunks, bm25 = _open(args)

    vocab = None
    if not args.no_check_oov:
        vocab = build_vocab_from_collection(collection)

    conv = Conversation(
        client, collection,
        name=args.index, strategy=args.strategy, top_k=args.top_k,
        chunks=chunks, bm25=bm25, vocab=vocab,
        enable_rewrite=not args.no_rewrite,
        guard_rewrite=not args.no_rewrite_guard,
    )

    print(f"\n{'─' * 70}")
    print(f"多轮对话   索引: {args.index}   策略: {args.strategy}")
    print(f"查询改写: {'开' if conv.enable_rewrite else '关'}   "
          f"改写校验: {'开' if vocab else '关'}")
    print(f"{'─' * 70}")
    print("输入问题回车；:history 看历史，:clear 清空，:exit 退出\n")

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in (":exit", ":q", ":quit"):
            break
        if line == ":clear":
            conv.clear()
            print("（历史已清空）\n")
            continue
        if line == ":history":
            if not conv.turns:
                print("（还没有历史）\n")
            for i, t in enumerate(conv.turns, 1):
                print(f"  [{i}] {t.question}")
                if t.rewritten:
                    print(f"       -> 改写为: {t.standalone}")
            print()
            continue

        # ── 检索 + 生成（流式）──
        # 注意：Conversation.ask 只 yield 片段，不负责打印 ——
        # 打印是 CLI 的事。库往 stdout 写东西会让它没法被别的程序复用。
        sys.stdout.write("\n助手 > ")
        sys.stdout.flush()
        for piece in conv.ask(line, stream=True):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        turn = conv.last

        # 把改写和告警展示出来 —— 这是这套实现最需要被看见的部分
        notes = []
        if turn.rewritten:
            notes.append(f"改写为「{turn.standalone}」")
        if turn.warning:
            notes.append(f"⚠️ {turn.warning}")
        if turn.oov:
            notes.append(f"⚠️ 改写引入了语料里没有的词：{', '.join(turn.oov)}")
        if notes:
            print("  " + "  ".join(notes))

        print(f"  来源: {', '.join(h['source'] for h in turn.hits[:3])}\n")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """启动 HTTP 服务。"""
    from .server import serve

    serve(host=args.host, port=args.port)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from evaluation.run_eval import main as eval_main

    _apply_user_dict(args)
    strategies = args.strategies or ["vector", "rerank"]
    return eval_main(
        strategies=strategies,
        runs=args.runs,
        index=args.index,
        eval_set_path=args.eval_set,
    )


# ══════════════════════════════════════════════════════
# 参数定义
# ══════════════════════════════════════════════════════


def _apply_user_dict(args: argparse.Namespace, *, verbose: bool = False) -> None:
    """加载 --dict 指定的自定义词典。

    ⚠️ 建索引和查询必须用【同一个】词库，否则查询和文档的切法对不上，
        BM25 会静默失效。所以这个参数在每个涉及分词的命令上都有。
    """
    path = getattr(args, "user_dict", None)
    if not path:
        return
    from .tokenize import load_user_dict

    n = load_user_dict(path)
    if verbose:
        print(f"已加载自定义词典：{n} 个词（{path}）")


def _add_index_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-i", "--index", default=COLLECTION_NAME,
        help=f"索引名（默认 {COLLECTION_NAME}）",
    )
    p.add_argument(
        "--dict", dest="user_dict", default=None, metavar="PATH",
        help="自定义分词词典（中文领域词）。建索引和查询要用同一份",
    )


def _add_strategy_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", default="rerank", choices=STRATEGY_CHOICES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my-rag",
        description="my-rag —— 手写的 RAG 检索工具。每个默认参数都有实验依据。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p = sub.add_parser("init", help="在当前目录生成 .ragignore 和 rag.toml 模板")
    p.add_argument("path", nargs="?", default=".", help="语料目录（默认当前目录）")
    p.add_argument("--force", action="store_true", help="覆盖已存在的配置文件")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("index", help="给文档目录建索引")
    p.add_argument("paths", nargs="+", help="文档目录或文件（md/txt/pdf/docx/代码）")
    p.add_argument(
        "-n", "--name", default=None,
        help=f"索引名（默认读 rag.toml，再退到 {COLLECTION_NAME}）",
    )
    p.add_argument(
        "--docs-src", default=None,
        help="可选：展开 Markdown 里指向该目录的代码引用指令（FastAPI 文档专用）",
    )
    p.add_argument(
        "--dict", dest="user_dict", default=None, metavar="PATH",
        help="自定义分词词典（中文领域词，如「年假」「带薪年休假」）",
    )
    p.add_argument(
        "--exclude", nargs="*", default=[], metavar="GLOB",
        help="排除的文件，如 --exclude '*.txt' 'CHANGELOG*'",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="全量重建（默认是增量：只向量化内容变了的块）",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="预演：只统计会索引什么，不调用 API、不写库",
    )
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.set_defaults(func=cmd_index)

    # ask
    p = sub.add_parser("ask", help="提问")
    p.add_argument("question", nargs="?", default=None, help="问题（用 --file 时可省略）")
    p.add_argument("-f", "--file", default=None, metavar="PATH",
                   help="从文件批量读问题（每行一个，# 开头为注释）")
    p.add_argument("--json", action="store_true",
                   help="输出 JSON（批量时为 JSONL，一行一条）")
    p.add_argument("--stream", action="store_true", help="流式输出答案")
    _add_index_arg(p)
    _add_strategy_arg(p)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--show-hits", action="store_true", help="同时打印材料来源")
    p.set_defaults(func=cmd_ask)

    # chat
    p = sub.add_parser("chat", help="多轮对话（追问可引用前文）")
    _add_index_arg(p)
    _add_strategy_arg(p)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--no-rewrite", action="store_true",
                   help="关掉追问改写（对照组：看改写到底有没有用）")
    p.add_argument("--no-rewrite-guard", action="store_true",
                   help="关掉改写护栏（模型编了语料外的词时不回退，只报警告）")
    p.add_argument("--no-check-oov", action="store_true",
                   help="关掉改写结果的术语校验（连同护栏一起失效）")
    p.set_defaults(func=cmd_chat)

    # retrieve
    p = sub.add_parser("retrieve", help="只看检索结果，不生成答案")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    _add_index_arg(p)
    _add_strategy_arg(p)
    p.add_argument("--top-k", type=int, default=4)
    p.set_defaults(func=cmd_retrieve)

    # trace
    p = sub.add_parser("trace", help="逐层打印检索管线（调检索用）")
    p.add_argument("question")
    _add_index_arg(p)
    _add_strategy_arg(p)
    p.add_argument("--answer", action="store_true", help="顺带生成答案")
    p.set_defaults(func=cmd_trace)

    # list / info / drop
    sub.add_parser("list", help="列出所有索引").set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="索引详情")
    _add_index_arg(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("drop", help="删除索引")
    p.add_argument("name")
    p.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    p.set_defaults(func=cmd_drop)

    # serve
    p = sub.add_parser("serve", help="启动 HTTP 服务 + 网页界面")
    p.add_argument("--host", default="127.0.0.1", help="监听地址（默认只监听本机）")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    # eval
    p = sub.add_parser("eval", help="跑评估")
    _add_index_arg(p)
    p.add_argument("--strategies", nargs="+", choices=STRATEGY_CHOICES)
    p.add_argument("--runs", type=int, default=1, help="每个策略重复几轮（测稳定性）")
    p.add_argument("--eval-set", default=None, help="自定义评估集 JSON 路径")
    p.set_defaults(func=cmd_eval)

    # 兼容旧用法
    p = sub.add_parser("build", help="[已弃用] 建内置 FastAPI 示例索引")
    p.set_defaults(func=cmd_build_compat)

    return parser


def cmd_build_compat(args: argparse.Namespace) -> int:
    from .ingest import build_knowledge_base

    print("[提示] build 是旧命令，等价于：")
    print("       my-rag index <示例语料> --name fastapi_docs --docs-src data/docs_src\n")
    build_knowledge_base()
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, FileNotFoundError, PermissionError) as exc:
        # 这些是【用户能自己解决】的错误：没配 Key、索引不存在、路径写错。
        # 打完整堆栈只会把那条真正有用的提示淹没在几十行里。
        # 真正的 bug（TypeError / KeyError 之类）不在这里接，让它正常爆出来。
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
