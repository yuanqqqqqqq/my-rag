"""下载语料：FastAPI 官方文档 + 代码示例。

    python scripts/fetch_data.py

放在 data/ 下，位置和 rag/config.py 里约定的保持一致：

    data/fastapi-repo/docs/en/docs/tutorial/   51 篇 markdown（主体语料）
    data/docs_src/                             464 个代码文件

为什么要跑这个脚本：
    这两份语料都是 FastAPI 官方仓库的内容，不是本项目的代码，
    所以没有随仓库提交 —— 仓库里只放工具本身和一个很小的中文示例
    （examples/hr-demo，开箱即可试）。

为什么 docs_src 要单独复制出来：
    文档里大量使用 {* ../../docs_src/X.py *} 这种引用指令指向代码示例。
    不把这些指令展开成真实代码，文档里 271 处「怎么做」的答案就是空壳 ——
    展开之后某道题的答对率从 0/5 变成 5/5，是语料侧最大的一次提升。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/fastapi/fastapi.git"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEST_REPO = DATA / "fastapi-repo"
DEST_SRC = DATA / "docs_src"


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    DATA.mkdir(exist_ok=True)

    # 1) 稀疏克隆：只取要用的两个目录，不拉整个仓库
    if DEST_REPO.exists():
        print(f"[1/2] {DEST_REPO.name}/ 已存在，跳过 clone")
    else:
        print("[1/2] 稀疏克隆 FastAPI 仓库（只取 docs）...")
        run(["git", "clone", "--depth", "1", "--filter=blob:none",
             "--sparse", REPO, str(DEST_REPO)])
        run(["git", "-C", str(DEST_REPO), "sparse-checkout", "set",
             "docs/en/docs", "docs/en/docs_src"])

    # 2) 把代码示例复制到约定位置
    src = DEST_REPO / "docs" / "en" / "docs_src"
    if not src.is_dir():
        print(f"\n[错误] 没找到 {src}")
        print("  试着删掉 data/fastapi-repo 后重跑本脚本。")
        return 1

    if DEST_SRC.exists():
        print(f"[2/2] {DEST_SRC.name}/ 已存在，跳过复制")
    else:
        print("[2/2] 复制 docs_src ...")
        shutil.copytree(src, DEST_SRC)

    docs = DEST_REPO / "docs" / "en" / "docs" / "tutorial"
    n_md = len(list(docs.glob("**/*.md")))
    n_src = len(list(DEST_SRC.glob("**/*")))

    print(f"\n完成：{n_md} 篇文档，{n_src} 个代码示例")
    print("下一步：")
    print("  cp .env.example .env   # 填入 ZHIPUAI_API_KEY")
    print("  python -m rag.cli build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
