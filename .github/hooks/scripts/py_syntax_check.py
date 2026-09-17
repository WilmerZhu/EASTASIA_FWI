#!/usr/bin/env python3
"""PostToolUse hook: 对刚编辑的 .py 文件做语法检查，失败时向 agent 回传提示。"""
import json
import py_compile
import re
import sys
from pathlib import Path

EDIT_TOOL = re.compile(r'edit|create|replace|insert|write', re.I)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if not EDIT_TOOL.search(str(payload.get('tool_name', ''))):
        return
    tool_input = payload.get('tool_input', {})
    candidates = []
    for k, v in tool_input.items():
        if isinstance(v, str) and 'path' in k.lower():
            candidates.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    candidates += [x for kk, x in item.items() if isinstance(x, str) and 'path' in kk.lower()]

    errors = []
    for p in dict.fromkeys(candidates):
        path = Path(p)
        if path.suffix == '.py' and path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(str(exc).strip())

    if errors:
        print(json.dumps({
            'systemMessage': '语法检查失败，请修复后继续:\n' + '\n'.join(errors)
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
