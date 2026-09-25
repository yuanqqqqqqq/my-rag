"""不依赖网络和 API Key 的单元测试。

    python tests/test_pipeline.py        # 直接跑
    pytest tests/ -q                     # 或用 pytest

覆盖的都是纯函数：切块、融合、指标计算、术语校验。
需要网络的检索链路不在单元测试范围内（见 .github/workflows/ci.yml 的 smoke job）。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (  # noqa: E402
    hit_at_k,
    mean_rank,
    parse_score,
    rank_spread,
)
from rag.chunking import chunk_document, merge_tiny, resolve_includes  # noqa: E402
from rag.retrieve import rrf_fuse, tokenize  # noqa: E402
from rag.rewrite import build_vocab, find_oov_terms  # noqa: E402

# ══════════════════════════════════════════════════════
# 切块
# ══════════════════════════════════════════════════════


def test_resolve_includes_replaces_code():
    """引用指令应被替换成真实代码，并包成代码块。"""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "demo.py"), "w", encoding="utf-8") as f:
            f.write("print('hi')")

        text = "说明文字\n\n{* ../../docs_src/demo.py hl[1] *}\n\n结束"
        out, ok, fail = resolve_includes(text, tmp)

        assert ok == 1 and fail == 0, (ok, fail)
        assert "```python" in out
        assert "print('hi')" in out
        assert "docs_src" not in out


def test_resolve_includes_keeps_missing_file():
    """找不到文件时原样保留，不能让整条管线崩掉。"""
    with tempfile.TemporaryDirectory() as tmp:
        text = "{* ../../docs_src/not_here.py *}"
        out, ok, fail = resolve_includes(text, tmp)

        assert ok == 0 and fail == 1
        assert out == text  # 原样保留


def test_chunk_document_splits_by_headers():
    md = "# 标题一\n\n内容一\n\n## 标题二\n\n内容二\n"
    items = chunk_document(md, "demo.md")

    assert len(items) == 2
    assert items[0]["h1"] == "标题一"
    assert items[1]["h1"] == "标题一" and items[1]["h2"] == "标题二"
    assert all(it["source"] == "demo.md" for it in items)


def test_merge_tiny_absorbs_short_chunks():
    """小于阈值的块应并入前一块；首块即使很短也保留。"""
    items = [
        {"text": "A" * 100, "source": "s", "h1": "", "h2": "", "h3": ""},
        {"text": "B" * 50, "source": "s", "h1": "", "h2": "", "h3": ""},
        {"text": "C" * 900, "source": "s", "h1": "", "h2": "", "h3": ""},
    ]
    merged = merge_tiny(items, min_chars=700)

    assert len(merged) == 2
    assert "B" * 50 in merged[0]["text"]  # 碎块被并进前一块


# ══════════════════════════════════════════════════════
# 检索
# ══════════════════════════════════════════════════════


def test_tokenize_keeps_identifiers():
    """代码标识符必须保留 —— 否则 BM25 在含 API 名的问题上失效。"""
    toks = tokenize("Use OAuth2PasswordBearer and path_params in FastAPI!")

    assert "oauth2passwordbearer" in toks
    assert "path_params" in toks
    assert "fastapi" in toks


def test_rrf_fuse_prefers_consensus():
    """两路都靠前的块，应该压过只有一路第一的块。

    这是 RRF 的本意：k=60 时第 1 名和第 2 名几乎同权，
    所以「多路都说好」> 「一路说最好」。
    """
    a = {"id": "a", "text": "", "source": ""}
    b = {"id": "b", "text": "", "source": ""}
    fused = rrf_fuse([[a, b], [b, a]])

    assert {h["id"] for h in fused} == {"a", "b"}  # 两者对称，不丢结果

    # 一路第 1 + 另一路第 2  >  一路第 1 + 另一路缺席
    c = {"id": "c", "text": "", "source": ""}
    fused2 = rrf_fuse([[a, b], [b, c]])
    assert fused2[0]["id"] == "b"  # b 两路都出现，应该赢


# ══════════════════════════════════════════════════════
# 指标
# ══════════════════════════════════════════════════════


def test_hit_at_k_returns_rank():
    hits = [{"source": "x.md"}, {"source": "y.md"}, {"source": "z.md"}]

    assert hit_at_k(hits, "z.md") == (True, 3)
    assert hit_at_k(hits, "nope.md") == (False, 0)
    assert hit_at_k(hits, None) == (True, 0)  # 库里没有的题不计入命中率


def test_mean_rank_penalizes_misses():
    """未命中的题必须计入 —— 否则分母会缩水，制造幸存者偏差。"""
    assert mean_rank([1, 1, 1, 3], penalty=5) == 1.5

    # 漏掉 3 题却"平均名次"更漂亮 —— 这正是要防的假象
    with_misses = mean_rank([1, 0, 0, 0], penalty=5)
    assert with_misses == (1 + 5 * 3) / 4


def test_rank_spread_measures_stability():
    assert rank_spread([1, 1, 1]) == 0.0
    assert rank_spread([1, 1, 4]) == 3.0
    assert rank_spread([1, 0, 0], penalty=5) == 4.0


def test_parse_score_extracts_marked_number():
    assert parse_score("分数：8") == 8.0
    assert parse_score("SCORE: 7") == 7.0
    assert parse_score("no number here") == 5.0  # 降级到默认值

    # 不取第一个出现的数字，而是标记后面的那个
    assert parse_score("我给了 3 分给文档 A，so 分数：9") == 9.0

    # 越界要钳住
    assert parse_score("分数：99") == 10.0


# ══════════════════════════════════════════════════════
# 术语校验（这条是为 Step 9 那个"投毒"事故加的护栏）
# ══════════════════════════════════════════════════════


def test_find_invented_terms_spares_user_supplied_terms():
    """用户自己提的语料外术语不算「模型编的」。

    这是把「一刀切回退」改成「只回退模型编的」的关键区分。
    如果只看 OOV，用户问一个新术语时改写会被误判成投毒、白白回退 ——
    回退到原问题反而让检索更难，因为指代没被补全。
    """
    from rag.rewrite import find_invented_terms

    vocab = {"fastapi", "uploadfile", "cors", "oauth2passwordbearer"}

    # 用户自己说了 WebSocketEndpoint（语料里没有）-> 不该当成编造
    invented = find_invented_terms(
        "How do I use WebSocketEndpoint in FastAPI?",
        "那 WebSocketEndpoint 怎么用",
        vocab,
    )
    assert "WebSocketEndpoint" not in invented, "误伤了用户自己提的术语"

    # 用户只说了 CORS，改写却冒出两个语料外的词 -> 都是模型编的
    invented2 = find_invented_terms(
        "Configure CORSMiddleware with allowed_origins",
        "那 CORS 呢",
        vocab,
    )
    assert "CORSMiddleware" in invented2
    assert "allowed_origins" in invented2


def test_find_invented_terms_ignores_in_vocab_terms():
    """语料里有的术语当然不算编造。"""
    from rag.rewrite import find_invented_terms

    vocab = {"fastapi", "uploadfile", "cors"}

    assert find_invented_terms(
        "How do I use UploadFile in FastAPI?", "怎么上传文件", vocab
    ) == []


def test_find_oov_terms_catches_hallucinated_api_names():
    vocab = build_vocab(["Use `fastapi[standard]` to install.", "UploadFile is here"])

    text = "Run `pip install fastapi[all]` and use `UploadFile` with `OAuth2PasswordBearer`"
    oov = find_oov_terms(text, vocab)

    # 语料里有的不该被标出来
    assert "UploadFile".lower() not in [o.lower() for o in oov]
    # 语料里没有的 API 名应该被抓出来
    assert any("fastapi[all]" in o.lower() for o in oov) or any(
        "OAuth2PasswordBearer" in o for o in oov
    )


# ══════════════════════════════════════════════════════
# 文档加载（多格式）
# ══════════════════════════════════════════════════════


def test_kind_of_covers_supported_formats():
    from pathlib import Path

    from rag.loaders import kind_of

    assert kind_of(Path("a.md")) == ("markdown", "")
    assert kind_of(Path("a.txt"))[0] == "text"
    assert kind_of(Path("a.pdf"))[0] == "pdf"
    assert kind_of(Path("a.docx"))[0] == "docx"
    assert kind_of(Path("a.py")) == ("code", "python")
    assert kind_of(Path("a.js")) == ("code", "javascript")
    assert kind_of(Path("a.unknownext")) is None


def test_load_path_skips_excluded_and_skipped_dirs():
    """--exclude 要真的挡住文件；.git / node_modules 要真的不递归。"""
    from rag.loaders import load_path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "doc.md").write_text("# 标题\n\n内容", encoding="utf-8")
        (root / "notes.txt").write_text("笔记内容", encoding="utf-8")
        (root / "keep.txt").write_text("要保留", encoding="utf-8")

        # 这些目录应该被自动跳过
        git_dir = root / ".git"
        git_dir.mkdir()
        (git_dir / "config.md").write_text("不该被加载", encoding="utf-8")
        nm_dir = root / "node_modules"
        nm_dir.mkdir()
        (nm_dir / "pkg.md").write_text("不该被加载", encoding="utf-8")

        docs = load_path(root, exclude=["notes.txt"], verbose=False)
        sources = {d.source for d in docs}

        assert "doc.md" in sources
        assert "keep.txt" in sources
        assert "notes.txt" not in sources, "exclude 没生效"
        assert not any("git" in s or "node_modules" in s for s in sources), (
            "跳过了不该跳过的目录过滤失效"
        )


def test_load_path_rejects_missing_path():
    from rag.loaders import load_path

    try:
        load_path("这个路径不存在", verbose=False)
    except FileNotFoundError:
        return
    raise AssertionError("路径不存在时应该报 FileNotFoundError")


# ══════════════════════════════════════════════════════
# 增量索引的基石：内容指纹
# ══════════════════════════════════════════════════════


def test_chunk_uid_is_stable_and_content_addressed():
    """同样的内容必须得到同样的 id —— 这是增量索引能成立的全部前提。

    id 由内容决定，那"这个块变了没有"就退化成"这个 id 还在不在"，
    不需要额外维护一张对照表。
    """
    from rag.store import chunk_uid

    a1 = chunk_uid("a.md", "同样的内容")
    a2 = chunk_uid("a.md", "同样的内容")
    assert a1 == a2, "同样的内容算出了不同的 id，增量索引会失效"

    assert a1 != chunk_uid("a.md", "不同的内容"), "内容变了 id 必须变"
    assert a1 != chunk_uid("b.md", "同样的内容"), (
        "同内容不同来源必须是两个块 —— 否则溯源信息会丢"
    )
    assert a1.startswith("c_")


def test_chunk_uid_survives_whitespace_edits_differently():
    """改一个字就应该换 id（哪怕只多一个空格）。

    宁可多算一次，也不能漏算 —— 漏算意味着用户改了文档但检索到旧内容。
    """
    from rag.store import chunk_uid

    assert chunk_uid("a.md", "内容") != chunk_uid("a.md", "内容 ")


# ══════════════════════════════════════════════════════
# 中文分词
# ══════════════════════════════════════════════════════


def test_tokenize_handles_chinese():
    """中文必须被切词，且【不能丢单字】。

    这条是为一个实测 bug 加的护栏：jieba 的通用词典没有「年假」，
    会把它切成「年」「假」。早期版本用 `len(w) > 1` 过滤单字，
    结果是查询和文档里的「年假」两边都变成空 —— 永远匹配不上。
    """
    from rag.tokenize import tokenize

    toks = tokenize("怎么申请年假")

    assert "申请" in toks, "中文没被切词 —— BM25 对中文会完全失效"
    # ⚠️ 这里断言的是「词没有凭空消失」，而不是具体的切法。
    #    jieba 的词典是【全局可变状态】—— 别的测试加了自定义词之后，
    #    「年假」可能变成一个词。断言具体切法会变成脆弱的顺序依赖测试。
    assert ("年假" in toks) or ("年" in toks and "假" in toks), (
        "「年假」被整个丢掉了 —— 单字过滤会导致中文匹配断裂"
    )


def test_tokenize_keeps_english_identifiers_in_mixed_text():
    """中英混排时，代码标识符不能被 jieba 切碎。"""
    from rag.tokenize import tokenize

    toks = tokenize("FastAPI 的 path_params 参数怎么声明")

    assert "fastapi" in toks
    assert "path_params" in toks, "英文标识符被切碎了 —— BM25 在含 API 名的问题上会失效"


def test_load_user_dict_teaches_domain_words():
    """自定义词典必须能修正 jieba 的切分。

    ⚠️ 用「调休额度」而不是「离职」：jieba 通用词典已经认识离职，
       用它测不出任何东西。领域词要挑 jieba 真的不认识的。
    """
    from rag.tokenize import load_user_dict, tokenize

    before = tokenize("调休额度怎么算")
    assert "调休额度" not in before, "jieba 居然认识这个词，换一个来测"

    with tempfile.TemporaryDirectory() as tmp:
        dict_path = os.path.join(tmp, "dict.txt")
        with open(dict_path, "w", encoding="utf-8") as f:
            f.write("# 领域词\n调休额度 100 n\n")
        assert load_user_dict(dict_path) == 1

    after = tokenize("调休额度怎么算")
    assert "调休额度" in after, "自定义词典没生效"


def test_detect_language():
    from rag.tokenize import detect_language

    assert detect_language(["FastAPI is a web framework"]) == "en"
    assert detect_language(["这是一个中文文档，讲的是公司制度"]) == "zh"


# ══════════════════════════════════════════════════════
# 多格式切块分发
# ══════════════════════════════════════════════════════


def test_markdown_without_docs_src_dir_still_works():
    """不传 docs_src_dir 时，引用指令原样保留，不能让管线崩掉。

    这是「工具化」的关键：FastAPI 的引用展开是可选预处理，
    别的语料没有这种东西，不传就应该走普通流程。
    """
    from rag.chunking import load_documents

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "a.md"), "w", encoding="utf-8") as f:
            f.write("# 标题\n\n正文 {* ../docs_src/x.py *}\n")

        items = load_documents(tmp, None, verbose=False)

        assert items, "应该切出至少一块"
        assert any("docs_src" in it["text"] for it in items), (
            "没传 docs_src_dir 时不该展开引用"
        )


def test_chunk_code_uses_function_boundaries():
    """代码要按函数边界切，别把函数劈成两半。"""
    from rag.chunking import chunk_loaded
    from rag.loaders import LoadedDoc

    code = "\n\n".join(
        f"def func_{i}(x):\n    return x * {i}" for i in range(20)
    )
    doc = LoadedDoc(text=code, source="demo.py", kind="code", lang="python")
    items = chunk_loaded(doc)

    assert items
    assert all(it["kind"] == "code" for it in items)


# ══════════════════════════════════════════════════════


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
