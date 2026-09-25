"""一键复现：从零跑到评估对比。

    python scripts/reproduce.py

步骤：
    1. 下载语料       （需要网络）
    2. 建向量库       （需要 ZHIPUAI_API_KEY，约 1 分钟）
    3. 跑单元测试     （不需要网络）
    4. 跑评估对比     （需要 Key，约 120 次 API 调用）

用 --skip-eval 可以跳过第 4 步（它最花时间）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(n: int, total: int, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"[{n}/{total}] {title}")
    print("=" * 70)


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[失败] {' '.join(cmd)} 返回 {result.returncode}")
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true", help="跳过语料下载")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估（省 API 调用）")
    args = parser.parse_args()

    total = 4
    py = sys.executable

    if not args.skip_fetch:
        step(1, total, "下载语料")
        run([py, "scripts/fetch_data.py"])
    else:
        step(1, total, "下载语料（已跳过）")

    step(2, total, "建向量库")
    run([py, "-m", "rag.cli", "build"])

    step(3, total, "单元测试")
    run([py, "tests/test_pipeline.py"])

    if args.skip_eval:
        step(4, total, "评估（已跳过）")
    else:
        step(4, total, "评估对比：纯向量 vs LLM 精排")
        run([py, "-m", "rag.cli", "eval", "--strategies", "vector", "rerank"])

    print(f"\n{'=' * 70}")
    print("完成。基线参考：纯向量 平均命中名次 2.14 / 命中 7/7；")
    print("      rerank   平均命中名次 1.50 / 检索相关性 7.3 -> 8.0")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
