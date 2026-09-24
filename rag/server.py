"""HTTP 服务：把 my-rag 暴露成一个 API。

    my-rag serve --port 8000

    GET  /              最小网页界面（直接就能用）
    GET  /health        健康检查
    GET  /indexes       列出所有索引
    POST /ask           {"question": "...", "index": "...", "strategy": "..."}
    POST /retrieve      同上，但不生成答案（便宜）

为什么单独一个模块：CLI 是给人用的，API 是给程序用的 ——
两者的错误处理、输出格式、并发模型都不一样，混在一起会互相牵扯。

⚠️ 这是个【本地工具的服务端】，没有鉴权、没有限流、默认只监听 127.0.0.1。
   别直接挂公网 —— 它背后的 LLM API 是按量计费的。
"""

import os
import time
from typing import Any

from .cli import STRATEGY_CHOICES
from .client import get_client
from .config import COLLECTION_NAME, FINAL_K
from .generate import generate_answer
from .retrieve import build_bm25, retrieve
from .store import get_collection, get_index_config, list_collections, load_chunks

# ⚠️ 这个文件【故意不写 from __future__ import annotations】。
#
# 那一行会把所有类型注解变成字符串，FastAPI 要靠 get_type_hints 把它们
# 解析回真正的类。如果请求模型定义在函数内部（局部名），解析就会失败 ——
# FastAPI 不会报错，而是【静默降级】把请求体当成 query 参数，
# 于是所有 POST 都返回 422「Field required」，且错误信息完全指不到真正的原因。
#
# 实测踩过这个坑。所以：请求模型定义在模块级，且不开 future annotations。


def _require_pydantic():
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "启动 HTTP 服务需要 fastapi 和 uvicorn：\n"
            "    pip install fastapi uvicorn\n"
            "  （或 pip install -e '.[serve]'）"
        ) from exc
    return BaseModel, Field


BaseModel, Field = _require_pydantic()


class AskRequest(BaseModel):
    """问答请求体。定义在模块级 —— 见文件头关于注解解析的说明。"""

    question: str = Field(..., min_length=1, description="用户问题")
    index: str = Field(COLLECTION_NAME, description="索引名")
    strategy: str = Field("rerank", description="检索策略")
    top_k: int = Field(FINAL_K, ge=1, le=20)

# 打开过的索引缓存：{index_name: (collection, chunks, bm25)}
# 为什么缓存：load_chunks + 建 BM25 索引在 534 块的库上要一两秒，
# 每个 HTTP 请求都重建一遍是纯浪费。
_CACHE: dict[str, tuple] = {}


def _open_index(name: str, strategy: str):
    key = f"{name}|{strategy}"
    if key in _CACHE:
        return _CACHE[key]

    # 自动应用建索引时记录的分词词典（否则 BM25 会静默失效）
    saved = get_index_config(name).get("dict")
    if saved and os.path.isfile(saved):
        from .tokenize import load_user_dict

        load_user_dict(saved)

    collection = get_collection(name)

    chunks = bm25 = None
    if strategy in ("hybrid", "hybrid-rerank"):
        chunks = load_chunks(collection)
        bm25 = build_bm25(chunks)

    _CACHE[key] = (collection, chunks, bm25)
    return _CACHE[key]


def _hits_payload(hits: list[dict]) -> list[dict]:
    out = []
    for i, h in enumerate(hits, 1):
        item: dict[str, Any] = {
            "rank": i,
            "source": h["source"],
            "text": h["text"],
        }
        for src, dst in (("dist", "distance"), ("llm_score", "llm_score"),
                         ("rrf", "rrf")):
            if src in h:
                item[dst] = round(float(h[src]), 6)
        out.append(item)
    return out


def build_app():
    """构造 FastAPI 应用。"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(
        title="my-rag",
        description="手写的 RAG 检索工具 —— 每个默认参数都有实验依据",
        version="0.2.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/indexes")
    def indexes() -> list[dict]:
        return list_collections()

    def _run(req: AskRequest, *, generate: bool) -> dict:
        if req.strategy not in STRATEGY_CHOICES:
            raise HTTPException(
                status_code=400,
                detail=f"未知策略 {req.strategy!r}，可选：{', '.join(STRATEGY_CHOICES)}",
            )

        try:
            client = get_client()
            collection, chunks, bm25 = _open_index(req.index, req.strategy)
        except (RuntimeError, FileNotFoundError) as exc:
            # 这些是配置类错误，用 400 告诉调用方"你传的东西我处理不了"，
            # 而不是 500 —— 500 会让人以为服务崩了
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        t0 = time.perf_counter()
        hits = retrieve(
            client, collection, req.question,
            strategy=req.strategy, top_k=req.top_k, chunks=chunks, bm25=bm25,
        )
        t_retrieval = time.perf_counter() - t0

        payload = {
            "question": req.question,
            "index": req.index,
            "strategy": req.strategy,
            "elapsed_seconds": {"retrieval": round(t_retrieval, 3)},
            "hits": _hits_payload(hits),
        }

        if generate:
            t1 = time.perf_counter()
            payload["answer"] = generate_answer(client, req.question, hits)
            payload["elapsed_seconds"]["generation"] = round(
                time.perf_counter() - t1, 3
            )

        return payload

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        return _run(req, generate=True)

    @app.post("/retrieve")
    def retrieve_only(req: AskRequest) -> dict:
        return _run(req, generate=False)

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return INDEX_HTML

    return app


# ══════════════════════════════════════════════════════
# 最小网页界面：单文件、零依赖、直接能跑
# ══════════════════════════════════════════════════════

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>my-rag</title>
<style>
  :root { --bg:#0f1115; --fg:#e6e6e6; --dim:#8b93a1; --line:#252a34; --accent:#4c9aff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.6 -apple-system,"Segoe UI","PingFang SC",sans-serif; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 24px; }
  .row { display:flex; gap:8px; margin-bottom:16px; flex-wrap: wrap; }
  input, select, button { font: inherit; padding:9px 12px; border-radius:8px;
                          border:1px solid var(--line); background:#171b22; color:var(--fg); }
  input[type=text] { flex:1; min-width:240px; }
  button { background:var(--accent); border-color:var(--accent); color:#fff;
           cursor:pointer; font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  .ans { background:#171b22; border:1px solid var(--line); border-radius:10px;
         padding:16px; white-space:pre-wrap; min-height:60px; }
  .hits { margin-top:18px; }
  .hit { border-top:1px solid var(--line); padding:10px 0; font-size:13px; }
  .src { color:var(--accent); font-weight:600; }
  .score { color:var(--dim); margin-left:8px; }
  .preview { color:var(--dim); margin-top:4px;
             display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
             overflow:hidden; }
  .meta { color:var(--dim); font-size:12px; margin-top:10px; }
  .err { color:#ff6b6b; }
</style>
</head>
<body>
<div class="wrap">
  <h1>my-rag</h1>
  <div class="sub">手写的 RAG 检索工具 · 答案带来源溯源</div>

  <div class="row">
    <select id="index"></select>
    <select id="strategy">
      <option value="rerank">rerank（精排，默认）</option>
      <option value="vector">vector（纯向量，最快）</option>
      <option value="hybrid">hybrid（向量+BM25）</option>
      <option value="hybrid-rerank">hybrid-rerank</option>
    </select>
  </div>

  <div class="row">
    <input id="q" type="text" placeholder="问点什么…（回车提交）">
    <button id="go">提问</button>
  </div>

  <div class="ans" id="ans">答案会显示在这里。</div>
  <div class="hits" id="hits"></div>
  <div class="meta" id="meta"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);

async function loadIndexes() {
  const sel = $('index');
  try {
    const list = await (await fetch('/indexes')).json();
    sel.innerHTML = '';
    if (!list.length) {
      sel.innerHTML = '<option>（还没有索引）</option>';
      return;
    }
    for (const it of list) {
      const o = document.createElement('option');
      o.value = it.name;
      o.textContent = `${it.name}  (${it.count} 块)`;
      sel.appendChild(o);
    }
  } catch (e) { sel.innerHTML = '<option>读取失败</option>'; }
}

async function ask() {
  const q = $('q').value.trim();
  if (!q) return;

  $('go').disabled = true;
  $('ans').textContent = '检索中…';
  $('ans').className = 'ans';
  $('hits').innerHTML = '';
  $('meta').textContent = '';

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question: q, index: $('index').value,
                             strategy: $('strategy').value })
    });
    const data = await res.json();
    if (!res.ok) { throw new Error(data.detail || res.statusText); }

    $('ans').textContent = data.answer;
    $('meta').textContent =
      `检索 ${data.elapsed_seconds.retrieval}s + 生成 ${data.elapsed_seconds.generation}s`;

    $('hits').innerHTML = data.hits.map(h => `
      <div class="hit">
        <span class="src">${h.rank}. ${h.source}</span>
        ${h.llm_score !== undefined ? `<span class="score">llm=${h.llm_score}</span>` : ''}
        ${h.distance !== undefined ? `<span class="score">dist=${h.distance}</span>` : ''}
        <div class="preview">${h.text.replace(/</g,'&lt;').slice(0, 200)}</div>
      </div>`).join('');
  } catch (e) {
    $('ans').textContent = '出错了：' + e.message;
    $('ans').className = 'ans err';
  } finally {
    $('go').disabled = false;
  }
}

$('go').onclick = ask;
$('q').addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
loadIndexes();
</script>
</body>
</html>
"""


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """启动服务。"""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "启动服务需要 uvicorn：pip install uvicorn"
        ) from exc

    print(f"\nmy-rag 服务启动中：http://{host}:{port}")
    print("  （本地工具，无鉴权，默认只监听本机）\n")
    uvicorn.run(build_app(), host=host, port=port, reload=reload, log_level="info")
