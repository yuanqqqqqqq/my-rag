"""向量库：多索引的建、查、列、删。

用 Chroma 做持久化向量库。注意 chroma_db/ 有 20M，已在 .gitignore 里 ——
它随时能从文档重建，不该进版本控制。

支持多个命名索引（collection），互不干扰：

    my-rag index ./my-docs --name mydocs
    my-rag ask "问题" --index mydocs
    my-rag list
"""

from __future__ import annotations

import hashlib
import json

import chromadb

from .client import embed
from .config import CHROMA_PATH, COLLECTION_NAME


def _validate_name(name: str) -> str:
    """Chroma 对集合名有长度和字符限制，提前拦下来给个能看懂的报错。

    提前拦的理由：Chroma 自己的报错是一串 pydantic 校验信息，
    对用户没有任何指导意义。
    """
    if not name or len(name) < 3:
        raise ValueError(
            f"索引名 {name!r} 太短（Chroma 要求 3~63 个字符）。\n"
            f"  换个长点的，比如 {name + '-docs'!r}"
        )
    if len(name) > 63:
        raise ValueError(f"索引名 {name!r} 太长（Chroma 最多 63 个字符）")
    cleaned = name.replace("-", "_").replace(".", "_")
    if not cleaned.replace("_", "").isalnum():
        raise ValueError(
            f"索引名 {name!r} 含有非法字符，只能包含字母、数字、下划线、连字符"
        )
    return cleaned


def open_db() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(name: str = COLLECTION_NAME):
    """打开指定索引；不存在时给出可执行下一步的报错。"""
    db = open_db()
    try:
        return db.get_collection(name=_validate_name(name))
    except ValueError:
        raise
    except Exception as exc:  # chroma 的异常类型跨版本不稳定，统一接住
        available = [c.name for c in db.list_collections()]
        hint = (
            f"现有索引：{', '.join(available)}"
            if available
            else "还没有任何索引，先建一个"
        )
        raise RuntimeError(
            f"索引 {name!r} 不存在。\n  {hint}\n"
            f"  建索引：my-rag index <文档目录> --name {name}"
        ) from exc


def chunk_uid(source: str, text: str) -> str:
    """块的内容指纹：source + 正文 的 sha1。

    为什么指纹里要带上 source：
        两篇文档里出现完全相同的一段话（版权声明、免责条款）很常见。
        只对正文取哈希的话，它们会被当成同一个块 —— 溯源信息就丢了。

    为什么用内容指纹当 id（而不是 chunk_0 / chunk_1）：
        这是【增量索引】的基础。id 由内容决定，
        那么"这个块变了没有"就退化成"这个 id 还在不在"，
        不需要额外维护一张对照表。
    """
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return "c_" + h.hexdigest()[:16]


def _metadata(it: dict) -> dict:
    return {
        "source": it["source"],
        "h1": it.get("h1", ""),
        "h2": it.get("h2", ""),
        "h3": it.get("h3", ""),
        "kind": it.get("kind", ""),
        "lang": it.get("lang", ""),
        "chunk_index": it.get("chunk_index", -1),
    }


def build_collection(
    client,
    items: list[dict],
    name: str = COLLECTION_NAME,
    *,
    rebuild: bool = False,
    config: dict | None = None,
    verbose: bool = True,
) -> dict:
    """建索引。默认【增量】：只向量化变化的块。

    为什么必须增量：改一个错别字就重算全库，等于每次编辑都付一次全量 API 费用。
    内容指纹（chunk_uid）让"哪些块变了"变成一个集合差运算 ——
    没变的块直接复用库里已有的向量，一次 embedding 调用都不用花。

    Args:
        rebuild: True 则丢掉旧库全量重建（切块参数改了的时候需要）

    Returns:
        {"added": 新增块数, "reused": 复用块数, "removed": 删除块数,
         "total": 总块数, "mode": "full" | "incremental"}
    """
    name = _validate_name(name)
    db = open_db()

    # 算指纹 + 按指纹去重（同一份内容重复出现只留一份）
    for it in items:
        it["uid"] = chunk_uid(it["source"], it["text"])
    items = list({it["uid"]: it for it in items}.values())
    for i, it in enumerate(items):
        it["chunk_index"] = i

    collection = None
    existing_ids: set[str] = set()

    if not rebuild:
        try:
            collection = db.get_collection(name=name)
            existing_ids = set(collection.get(include=[])["ids"])
        except Exception:
            collection = None

    # 旧版本用 chunk_0 / chunk_1 当 id，新版本用内容哈希当 id。
    # 混在一起没法比对，索性整体重建 —— 而且要让用户知道为什么，
    # 否则他会看到"明明没改文档，却重新向量化了全部"而一头雾水。
    legacy = {i for i in existing_ids if not i.startswith("c_")}
    if legacy and not rebuild:
        if verbose:
            print(
                f"检测到旧格式索引（{len(legacy)} 个旧 id，来自更早的版本）。\n"
                f"  id 方案已升级为内容哈希，本次将全量重建 —— 只需发生这一次。"
            )
        collection = None
        existing_ids = set()

    # 全新索引（或强制重建）：全部要向量化
    if collection is None:
        if verbose and rebuild:
            print("全量重建索引...")
        elif verbose:
            print("新建索引...")
        try:
            db.delete_collection(name)
        except Exception:
            pass
        collection = db.get_or_create_collection(name=name)
        new_items, removed = items, []
        mode = "full"
    else:
        new_items = [it for it in items if it["uid"] not in existing_ids]
        removed = sorted(existing_ids - {it["uid"] for it in items})
        mode = "incremental"

    # 只对新增/变化的块调 API —— 这是增量的全部意义
    if new_items:
        if verbose:
            if mode == "full":
                print(f"开始向量化 {len(new_items)} 块...")
            else:
                print(f"新增/变化 {len(new_items)} 块，开始向量化（其余复用已有向量）...")
        vectors = embed(client, [it["text"] for it in new_items])
        collection.add(
            documents=[it["text"] for it in new_items],
            embeddings=vectors,
            ids=[it["uid"] for it in new_items],
            metadatas=[_metadata(it) for it in new_items],
        )

    if removed:
        collection.delete(ids=removed)

    # 把这次用的参数记进索引 —— 下次 ask 时能自动读回来（尤其是分词词典）
    if config:
        try:
            set_index_config(name, config, collection=collection)
        except Exception as exc:  # noqa: BLE001 - 记配置失败不该让建库失败
            if verbose:
                print(f"  [警告] 索引配置写入失败（不影响检索）：{exc}")

    stats = {
        "added": len(new_items),
        "reused": len(items) - len(new_items),
        "removed": len(removed),
        "total": collection.count(),
        "mode": mode,
    }

    if verbose:
        if mode == "incremental":
            print(
                f"增量更新完成：新增 {stats['added']}，复用 {stats['reused']}，"
                f"删除 {stats['removed']} —— 现共 {stats['total']} 块"
            )
        else:
            print(f"建索引完成：{stats['total']} 块")

    return stats


# ══════════════════════════════════════════════════════
# 索引配置：把建索引时的参数存进索引本身
# ══════════════════════════════════════════════════════

CONFIG_PREFIX = "rag_"


def set_index_config(name: str, config: dict, *, collection=None) -> None:
    """把建索引时的参数写进 collection metadata。

    为什么重要：分词词典必须【建索引和查询时一致】，否则查询和文档的切法对不上，
    BM25 会静默失效 —— 不报错，只是效果悄悄变差。
    靠用户每次记得传 `--dict` 是靠不住的，所以把参数存进索引、查的时候自动读回来。

    ⚠️ Chroma 的 modify() 是【整体替换】metadata，不是合并 ——
       所以必须先读出来再合并写回，否则会把别的键抹掉。
    """
    collection = collection if collection is not None else get_collection(name)
    merged = dict(collection.metadata or {})
    for key, value in config.items():
        if value is None:
            continue
        # Chroma 的 metadata 只接受 str / int / float / bool
        merged[f"{CONFIG_PREFIX}{key}"] = (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        )
    collection.modify(metadata=merged)


def get_index_config(name: str) -> dict:
    """读回索引配置。没有配置过的索引返回空字典。"""
    try:
        collection = get_collection(name)
    except RuntimeError:
        return {}

    out: dict = {}
    for key, value in (collection.metadata or {}).items():
        if not key.startswith(CONFIG_PREFIX):
            continue
        short = key[len(CONFIG_PREFIX):]
        if isinstance(value, str) and value.startswith(("[", "{")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        out[short] = value
    return out


def load_chunks(collection) -> list[dict]:
    """把整库读进内存，供 BM25 建索引等本地操作使用。"""
    data = collection.get(include=["documents", "metadatas"])
    return [
        {"id": did, "text": doc, "source": meta["source"]}
        for did, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]


def list_collections() -> list[dict]:
    """列出所有索引及块数。"""
    db = open_db()
    out = []
    for c in db.list_collections():
        try:
            count = db.get_collection(name=c.name).count()
        except Exception:
            count = -1
        out.append({"name": c.name, "count": count})
    return sorted(out, key=lambda x: x["name"])


def drop_collection(name: str) -> bool:
    """删除索引。返回是否真的删掉了。"""
    db = open_db()
    try:
        db.delete_collection(_validate_name(name))
        return True
    except Exception:
        return False


def collection_info(name: str = COLLECTION_NAME) -> dict:
    """索引详情：块数、来源文件数、文档类型分布。"""
    collection = get_collection(name)
    data = collection.get(include=["metadatas"])

    sources: set[str] = set()
    kinds: dict[str, int] = {}
    for m in data["metadatas"]:
        sources.add(m.get("source", ""))
        k = m.get("kind", "") or "unknown"
        kinds[k] = kinds.get(k, 0) + 1

    return {
        "name": name,
        "chunks": collection.count(),
        "sources": len(sources),
        "kinds": kinds,
    }
