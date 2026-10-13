"""Verify this research archive without market data or third-party packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    checked = optional = 0
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Archive path escapes folder: {item['path']}")
        if item["scope"] == "local_only" and not path.exists():
            optional += 1
            continue
        if not path.is_file():
            raise ValueError(f"Missing file: {item['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"File differs from archived copy: {item['path']}")
        checked += 1
    links = 0
    documents = [ROOT / "README.md", *(ROOT / "reports").rglob("*.md")]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            if re.match(r"^[A-Za-z]:", target) or target.startswith(("/", "file:")):
                raise ValueError(f"Non-portable link in {path.name}: {target}")
            local = (path.parent / unquote(target.split("#")[0])).resolve()
            if not local.is_relative_to(ROOT) or not local.exists():
                raise ValueError(f"Broken archive link in {path.name}: {target}")
            links += 1
    nested_git = [p for p in ROOT.rglob(".git")]
    if nested_git:
        raise ValueError("A nested Git repository was found")
    print(f"OK: {checked} file hashes, {links} relative links; {optional} optional local panels absent.")


if __name__ == "__main__":
    main()
