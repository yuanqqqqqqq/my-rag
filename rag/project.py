"""项目配置：`.ragignore` 与 `rag.toml`。

这两个文件让 `my-rag index .` 零参数就能用：

    .ragignore          排除规则，一行一个 glob（语法同 .gitignore）
    rag.toml            索引参数

但**更重要的东西不在这里** —— 见 store.py 的 set_index_config：
建索引时的参数会被写进索引本身，ask 时自动读回来。

为什么最后这点最关键：
    分词词典必须【建索引和查询时一致】，否则查询和文档的切法对不上，
    BM25 会**静默失效** —— 不报错，只是效果悄悄变差。
    靠用户每次记得加 `--dict` 是靠不住的。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

RAGIGNORE_NAME = ".ragignore"
RAGTOML_NAME = "rag.toml"


@dataclass
class ProjectConfig:
    name: str | None = None          # 索引名
    exclude: list[str] = field(default_factory=list)
    dict_path: str | None = None     # 分词词典
    docs_src: str | None = None      # FastAPI 文档专用

    def is_empty(self) -> bool:
        return not any([self.name, self.exclude, self.dict_path, self.docs_src])


def load_ragignore(corpus_dir: Path) -> list[str]:
    """读取 .ragignore。格式同 .gitignore：一行一个 glob，# 是注释。

    为什么要有它：一次 `my-rag index .` 常常把 .log、草稿、备份文件
    一起塞进语料，稀释检索质量。写一次规则，之后每次都自动生效。
    """
    path = corpus_dir / RAGIGNORE_NAME
    if not path.is_file():
        return []

    patterns: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def load_rag_toml(corpus_dir: Path) -> ProjectConfig:
    """读取 rag.toml。文件不存在或格式错误都返回空配置，不阻断流程。"""
    path = corpus_dir / RAGTOML_NAME
    if not path.is_file():
        return ProjectConfig()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"  [警告] {RAGTOML_NAME} 解析失败，已忽略：{exc}")
        return ProjectConfig()

    exclude = data.get("exclude", [])
    if isinstance(exclude, str):        # 允许写成单个字符串
        exclude = [exclude]

    return ProjectConfig(
        name=data.get("name"),
        exclude=[str(p) for p in exclude],
        dict_path=data.get("dict"),
        docs_src=data.get("docs_src"),
    )


def load_project_config(corpus_dir: Path) -> ProjectConfig:
    """合并 rag.toml 与 .ragignore（两者是叠加关系，不互相覆盖）。"""
    config = load_rag_toml(corpus_dir)
    config.exclude = list(config.exclude) + load_ragignore(corpus_dir)
    return config


def write_ragignore_template(corpus_dir: Path, patterns: list[str]) -> Path:
    """生成一个 .ragignore 模板，供 `my-rag init` 使用。"""
    path = corpus_dir / RAGIGNORE_NAME
    body = [
        "# my-rag 排除规则（语法同 .gitignore，一行一个 glob）",
        "# 这些文件不会进入语料。",
        "",
    ]
    body += patterns
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path
