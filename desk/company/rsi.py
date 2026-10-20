"""Closed-loop Recursive Self-Improvement (RSI) engine for Desk OS v3.

Body is stored in desk/company/_rsi_chunks/*.part and assembled at import time
so the source can be pushed through size-limited GitHub APIs without truncation.
"""
from __future__ import annotations

from pathlib import Path as _Path

_chunks_dir = _Path(__file__).resolve().parent / "_rsi_chunks"
_src = "".join(p.read_text(encoding="utf-8") for p in sorted(_chunks_dir.glob("*.part")))
exec(compile(_src, str(_Path(__file__).resolve()), "exec"), globals())
del _Path, _chunks_dir, _src
