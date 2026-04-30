"""Pytest import setup.

中文说明: 确保从任意 pytest rootdir 启动时都能导入仓库内 forensic_agent 包。
English: Ensures the in-repo forensic_agent package can be imported regardless
of pytest rootdir discovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
