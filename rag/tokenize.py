"""分词：中英文自动识别。

BM25 是关键词检索 —— 它靠"查询词在文档里出现了几次"打分。
所以【分词方式直接决定 BM25 能不能用】。

原来的实现是英文正则，中文文档进去会变成这样：

    "怎么申请年假"  ->  ['怎么申请年假']    # 整句一个 token，永远匹配不上

结果就是中文语料的 BM25 完全失效 —— 而且【不报错】，只是效果静默变差。

现在的策略是**中英分开处理**：
    - 英文/代码标识符：走正则，保证 `OAuth2PasswordBearer` 不被切碎
    - 中文：走 jieba，按词切

为什么不用 jieba 处理全部：它会把 `path_params` 切成 `path` / `_` / `params`，
代码标识符一碎，BM25 在"含 API 名的问题"上就失效了 —— 而那恰恰是它最擅长的地方。
"""

from __future__ import annotations

import os
import re

_EN_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_HAS_CHINESE = re.compile(r"[一-鿿]")
_CHINESE_RUN = re.compile(r"[一-鿿]+")

# 一趟扫描用：英文/数字 或 一段连续中文，按出现顺序交替匹配
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]+")

_jieba = None
_dict_loaded = False


def _load_dict_file(path: str) -> int:
    """把词典文件的词灌进 jieba。调用前必须先拿到 jieba 实例。"""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                _jieba.add_word(line.split()[0])
                n += 1
    return n


def _get_jieba():
    """懒加载 jieba。

    首次导入要加载词典（约 1 秒），所以不在模块顶层 import ——
    否则每次跑纯英文的脚本都要白等这一秒。
    """
    global _jieba, _dict_loaded

    if _jieba is None:
        import jieba

        jieba.setLogLevel(60)  # 关掉 "Building prefix dict..." 那堆日志
        _jieba = jieba

    # 环境变量指定的自定义词典，自动加载一次。
    # 为什么用环境变量：建索引和查询是两个进程，
    # 两边必须用【同一个词典】分词，否则查询和文档的切法对不上。
    if not _dict_loaded:
        _dict_loaded = True
        env_path = os.getenv("MY_RAG_DICT")
        if env_path and os.path.isfile(env_path):
            _load_dict_file(env_path)

    return _jieba


def has_chinese(text: str) -> bool:
    return bool(_HAS_CHINESE.search(text))


def load_user_dict(path: str | os.PathLike) -> int:
    """加载自定义词典，教 jieba 认识领域词。

    **这是中文检索最重要的一个开关。** jieba 的通用词典里没有领域词，
    实测它会把这些词切碎：

        年假   ->  年 / 假          （公司制度类文档的必备词）
        离职   ->  离 / 职

    切碎之后，查询写「年假」和文档里的「年假」都会变成两个单字，
    **匹配虽然还能发生，但语义单元没了，BM25 的区分度会下降。**

    词典文件格式（每行一个词，可选 词频 词性）：
        年假 100 n
        带薪年休假 50 n
    """
    global _dict_loaded

    _get_jieba()
    n = _load_dict_file(str(path))
    _dict_loaded = True  # 别再让环境变量那份覆盖掉显式传进来的
    return n


def tokenize(text: str) -> list[str]:
    """中英混合分词。返回小写 token 列表，**保持原文顺序**。

    ⚠️ 这里**故意不过滤单字，也不用停用词表**，原因有两个：

    1. jieba 切碎领域词的兜底。它把「年假」切成「年」「假」时，
       保留单字至少还能让查询和文档在字级别匹配上；
       一旦过滤掉单字，匹配就彻底断了。

    2. 高频虚词不用手工过滤 —— **BM25 的 IDF 机制本来就会给
       处处都有的词很低的权重**。手写停用词表是重复劳动，还容易误伤。

    ⚠️ 实现上必须【一趟扫描】。早先的版本分两趟（先用正则抽所有英文数字、
       再抽所有中文段），结果是 token 顺序被打乱：

           原文: 工龄满 1 年不满 3 年
           旧结果: ['1', '3', '5', '工龄', '满', '年', '不满', ...]
                    ^^^^ 数字全跑到前面，原本不相邻的「满」和「年」挨到了一起

       对 BM25 没影响（它只统计词频，不看顺序），**但对任何顺序敏感的
       分析都是错的** —— 领域词检测就被这个坑过（把「满年」当成了候选词）。
    """
    tokens: list[str] = []

    for match in _TOKEN.finditer(text):
        piece = match.group(0)
        if _HAS_CHINESE.search(piece):
            jieba = _get_jieba()
            tokens.extend(w for w in jieba.lcut(piece) if w.strip())
        else:
            tokens.append(piece.lower())

    return tokens


# ══════════════════════════════════════════════════════
# 领域词检测：主动发现「被 jieba 切碎」的词
# ══════════════════════════════════════════════════════
#
# 为什么需要主动检测：
#     中文没有空格，分词器把词切错了你【看不出来】。
#     用户看到的不是报错，而是"检索效果莫名变差" ——
#     然后他会去怀疑 embedding、怀疑切块、怀疑模型，
#     唯独不会想到是分词的锅。这类问题必须由工具主动暴露出来。

_ZH_SINGLE = re.compile(r"^[一-鿿]$")

# 单字虚词：它们相邻是语法现象，不构成领域词。
# 不过滤的话，"的是""了的"这类会淹没真正的候选。
_ZH_FUNCTION_CHARS = set(
    "的了是在和与或等这那我你他她它们有为对从到把被就都而及其之也很会能可要没个"
    "些呢吗吧啊着过并但则于以由让使给地得上下里外前后时当只又再更最如若因所"
)


def detect_split_terms(
    texts: list[str], *, min_count: int = 3, top_n: int = 15
) -> list[tuple[str, int]]:
    """找出可能被 jieba 切碎的领域词。返回 [(候选词, 出现次数)]。

    做法：统计**相邻两个单字 token** 的共现次数。
        「年」和「假」如果总是挨在一起出现，那「年假」大概率是一个词。

    这是个启发式，会有误报（把「假使」这类拆错的也算上），
    所以只用来【提示用户配词典】，不自动改任何东西。

    已经配过词典的词不会被检测出来 —— 它们不再被切碎。
    """
    from collections import Counter

    pairs: Counter = Counter()
    for text in texts:
        toks = tokenize(text)
        single = [
            t if _ZH_SINGLE.match(t) and t not in _ZH_FUNCTION_CHARS else None
            for t in toks
        ]
        for a, b in zip(single, single[1:]):
            if a and b:
                pairs[a + b] += 1

    return [(w, n) for w, n in pairs.most_common(top_n) if n >= min_count]


def detect_language(texts: list[str], sample: int = 200) -> str:
    """粗略判断语料主语言。返回 'zh' / 'en' / 'mixed'。

    只看前 sample 个文件，够用了 —— 这里只需要给用户一个提示，
    不需要精确。
    """
    zh = en = 0
    for t in texts[:sample]:
        zh += len(_CHINESE_RUN.findall(t[:2000]))
        en += len(_EN_TOKEN.findall(t[:2000]))

    if zh == 0:
        return "en"
    if en == 0:
        return "zh"
    # 中文一个字算一个 token，英文一个词算一个，粗略按 3:1 折算
    return "zh" if zh * 3 > en else "mixed" if zh > 0 else "en"
