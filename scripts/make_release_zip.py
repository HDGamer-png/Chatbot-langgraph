import os, zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
release_dir = root / 'releases'
release_dir.mkdir(exist_ok=True)

exclude_dirs = {'.git', 'chat_history', 'static/uploads', 'releases', '__pycache__'}
exclude_patterns = {'.pyc', '.bak', '.env', '.DS_Store'}

zip_path = release_dir / 'Chatbot-langgraph-v3.1.0.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        # Skip excluded dirs
        if any(part in exclude_dirs for part in rel.parts):
            continue
        # Skip files by pattern
        if p.suffix in exclude_patterns:
            continue
        if p.is_dir():
            continue
        z.write(p, rel)

print(f'Created {zip_path}')
