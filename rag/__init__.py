"""my-rag —— 一个手写的 RAG 检索管线。

不是 LangChain 的封装，而是从零实现的检索链路。
**每个默认参数都是用对照实验测出来的**，不是拍脑袋定的 ——
这一点在下面的模块注释里随处可见，每条都带着当时的实测数据。

顶层用法：
    from rag.ingest import build_index
    from rag.retrieve import retrieve
"""

__version__ = "0.1.0"
