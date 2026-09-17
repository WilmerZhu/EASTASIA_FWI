#!/usr/bin/env python3
"""PreToolUse hook: 拒绝 agent 直接写入/删除不入库的数据目录。"""
import json
import re
import sys

PROTECTED = ('data', 'results', 'figures', 'output', 'logs', 'backup')
EDIT_TOOL = re.compile(r'edit|create|replace|insert|write|notebook|rename|move|delete', re.I)
DANGEROUS_RM = re.compile(r'\brm\s+(-[a-zA-Z]*r[a-zA-Z]*|--recursive)\b')
PROTECTED_RE = re.compile(r'(?:^|[\s/\'"=:])(?:' + '|'.join(PROTECTED) + r')/')


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _touches_protected(text: str) -> bool:
    return PROTECTED_RE.search(text.replace('\\', '/')) is not None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    tool = str(payload.get('tool_name', ''))
    tool_input = payload.get('tool_input', {})
    reason = None
    decision = 'deny'

    if EDIT_TOOL.search(tool):
        # 只检查路径类字段，避免误伤文件内容中提到的目录名
        paths = [v for k, v in tool_input.items() if isinstance(v, str) and 'path' in k.lower()]
        if any(_touches_protected(p) for p in paths):
            reason = f'{tool} 试图写入受保护目录 {PROTECTED}，请让用户手动操作'
    else:
        for s in _strings(tool_input):
            if DANGEROUS_RM.search(s) and _touches_protected(s):
                reason = '命令同时含递归删除与受保护数据目录，需用户确认'
                decision = 'ask'
                break

    if reason:
        print(json.dumps({
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': decision,
                'permissionDecisionReason': reason,
            }
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
