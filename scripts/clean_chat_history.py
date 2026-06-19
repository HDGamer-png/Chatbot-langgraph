#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean chat_history JSON files by removing <strong>/<b> tags, escaped forms, and markdown **bold** markers.
Creates a .bak copy for each file before modifying.
"""
import re
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAT_DIR = REPO_ROOT / "chat_history"


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    # remove raw HTML bold tags
    s2 = re.sub(r'</?(?:strong|b)[^>]*>', '', s, flags=re.IGNORECASE)
    # remove escaped forms like &lt;strong&gt;
    s2 = re.sub(r'&lt;/?(?:strong|b)[^&]*&gt;', '', s2, flags=re.IGNORECASE)
    # remove markdown bold **...**
    s2 = re.sub(r'\*\*(.*?)\*\*', r'\1', s2, flags=re.DOTALL)
    # remove stray **
    s2 = s2.replace('**', '')
    return s2


def process_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"SKIP (invalid json): {path.name} ({e})")
        return False

    turns = data.get('turns') or []
    changed = False
    for t in turns:
        if not isinstance(t, dict):
            continue
        fa = t.get('final_answer')
        if isinstance(fa, str):
            cleaned = clean_text(fa)
            if cleaned != fa:
                t['final_answer'] = cleaned
                changed = True

    if changed:
        bak = path.with_suffix(path.suffix + '.bak')
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Cleaned: {path.name}")
    else:
        print(f"No changes: {path.name}")
    return changed


def main():
    if not CHAT_DIR.exists():
        print(f"chat_history folder not found at {CHAT_DIR}")
        return
    files = sorted(CHAT_DIR.glob('*.json'))
    if not files:
        print("No JSON files found in chat_history/")
        return
    total = 0
    for f in files:
        if process_file(f):
            total += 1
    print(f"Done. Files modified: {total}/{len(files)}")


if __name__ == '__main__':
    main()
