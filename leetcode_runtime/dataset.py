"""Load the LeetCode execution records used by the repeat experiment."""
from __future__ import annotations

import json

from hgre_core.paths import LEETCODE_DATA


def load_problems():
    rows = []
    with open(LEETCODE_DATA, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["question_id"]))
    return rows


def build_script(problem, code):
    prompt = problem.get("prompt") or ""
    tests = problem.get("test") or ""
    entry = problem.get("entry_point") or ""
    parts = [prompt.rstrip(), code.rstrip()]
    if tests.strip():
        parts.append(tests.rstrip())
    if entry:
        parts.append("check(%s)" % entry)
    return "\n".join(parts) + "\n"
