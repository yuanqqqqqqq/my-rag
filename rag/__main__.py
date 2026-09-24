"""让 `python -m rag` 等价于 `python -m rag.cli`。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
