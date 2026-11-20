"""
Prompt lists for Photobash, Ideas and Rewrite - one writer with the same options Generate's Save tab uses
(Nov 15: shared list settings). Format lines | blocks | jsonl, append to an existing list or make a new numbered file,
skip prompts already in the list, optional .meta.jsonl sidecar. Every saved prompt also goes to the provenance index.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

from . import config
from .outputs import unique_path
from .store import append_index


def one_line(text: str) -> str:
    return " ".join((text or "").split())


def _path(settings: Dict[str, Any], name: str, stem: str, fmt: str) -> str:
    ext = ".jsonl" if fmt == "jsonl" else ".txt"
    base = (name or "").strip() or f"{stem}_{_dt.date.today().strftime('%Y%m%d')}"
    for e in (".txt", ".jsonl"):
        if base.lower().endswith(e):
            base = base[: -len(e)]
    return (base if os.path.isabs(base) else os.path.join(settings["group_dir"], base)) + ext


def _existing(path: str, fmt: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    if fmt == "jsonl":
        out = []
        for line in raw.splitlines():
            try:
                out.append(one_line(json.loads(line).get("prompt", "")))
            except ValueError:
                continue
        return [p for p in out if p]
    if fmt == "blocks":
        out = []
        for block in raw.replace("\r\n", "\n").split("\n\n"):
            lines = [l for l in block.split("\n") if l.strip()]
            if lines:
                out.append(one_line(" ".join(lines[1:] if len(lines) > 1 and lines[0].startswith("#") else lines)))
        return out
    return [one_line(l) for l in raw.splitlines() if l.strip()]


def peek(settings: Dict[str, Any], name: str, stem: str, fmt: str) -> Dict[str, Any]:
    path = _path(settings, name, stem, fmt)
    return {"path": path, "file": os.path.basename(path), "exists": os.path.isfile(path),
            "entries": len(_existing(path, fmt))}


def save_list(settings: Dict[str, Any], entries: List[Dict[str, Any]], name: str, stem: str, fmt: str = "lines",
              on_exists: str = "append", dedupe: bool = True, sidecar: bool = True) -> Dict[str, Any]:
    """entries: [{"text", "header", "meta": {...}, "index": {...}}] -> one list file (+ sidecar, provenance)."""
    if not entries:
        raise ValueError("nothing to save")
    fmt = fmt if fmt in ("lines", "blocks", "jsonl") else "lines"
    path = _path(settings, name, stem, fmt)
    exists = os.path.isfile(path)
    if exists and on_exists == "new":
        path, exists = unique_path(path), False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = set(_existing(path, fmt)) if (exists and dedupe) else set()
    from .outputs import prompt_sha
    from .prompt_grades import duff_shas
    duff = duff_shas()
    skipped_duff = 0
    start = len(_existing(path, fmt)) if exists else 0
    made = _dt.datetime.now().isoformat(timespec="seconds")
    chunks, metas, index, skipped = [], [], [], 0
    for e in entries:
        text = one_line(e["text"])
        if not text:
            continue
        if prompt_sha(text) in duff:                  # graded duff: kept in the archive, never in new lists
            skipped_duff += 1
            continue
        if dedupe and text in seen:
            skipped += 1
            continue
        seen.add(text)
        n = start + len(chunks) + 1
        meta = {"line": n, **(e.get("meta") or {}), "prompt_sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]}
        if fmt == "jsonl":
            chunks.append(json.dumps({"prompt": text, **(e.get("meta") or {})}, ensure_ascii=False))
        elif fmt == "blocks":
            chunks.append(f"# {e.get('header') or n}\n{text}")
        else:
            chunks.append(text)
        metas.append(meta)
        index.append({"file": path, "prompt": text, "made": made, **(e.get("index") or {})})
    if chunks:
        sep = "\n\n" if fmt == "blocks" else "\n"
        with open(path, "a" if exists else "w", encoding="utf-8", newline="\n") as h:
            if exists and fmt == "blocks":
                h.write("\n")
            h.write(sep.join(chunks) + "\n")
        if sidecar:
            sp = os.path.splitext(path)[0] + ".meta.jsonl"
            with open(sp, "a" if os.path.isfile(sp) else "w", encoding="utf-8", newline="\n") as h:
                for m in metas:
                    h.write(json.dumps(m, ensure_ascii=False) + "\n")
        append_index(config.RUNS_DIR, index)
    return {"path": path, "file": os.path.basename(path), "added": len(chunks), "skipped": skipped,
            "skipped_duff": skipped_duff, "action": "appended" if exists else "new"}


# ----------------------------------------------------------------------------------------------- per mode
def photobash_entries(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        imgs, w = r.get("images") or [], r.get("weights") or []
        main = max(range(len(imgs)), key=lambda i: float(w[i] or 0)) if imgs and len(w) == len(imgs) else 0
        out.append({"text": r["prompt"], "header": f"photobash {r.get('mode', '')} - " + " + ".join(os.path.basename(i) for i in imgs),
                    "meta": {"variant": f"photobash {r.get('mode', '')}", "preset": r.get("preset", ""), "model": r.get("model", ""),
                             "images": imgs, "weights": w, "takes": r.get("takes", []), "assignment": r.get("assignment", {})},
                    "index": {"mode": "photobash", "image": imgs[main] if imgs else "", "model": r.get("model", ""),
                              "preset": r.get("preset", ""), "instr": "", "variant": f"photobash {r.get('mode', '')}",
                              "sources": imgs, "weights": w, "takes": r.get("takes", []),
                              "assignment": r.get("assignment", {})}})
    return out


def ideas_entries(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"text": r["prompt"], "header": f"ideas {r.get('mode', '')}",
             "meta": {"variant": f"ideas {r.get('mode', '')}", "preset": r.get("preset", ""), "model": r.get("model", ""),
                      "roll": r.get("roll"), "filled": r.get("filled")},
             "index": {"mode": "ideas", "image": "", "model": r.get("model", ""), "preset": r.get("preset", ""),
                       "instr": "", "variant": f"ideas {r.get('mode', '')}"}} for r in results]


def rewrite_entries(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"text": r["prompt"], "header": f"rewrite {r.get('op', '')}" + (f" - {os.path.basename(r['image'])}" if r.get("image") else ""),
             "meta": {"variant": f"rewrite {r.get('op', '')}", "preset": r.get("preset", ""), "model": r.get("model", ""),
                      "source_prompt": r.get("src", ""), "image": r.get("image", ""), "flags": r.get("flags", [])},
             "index": {"mode": "rewrite", "image": r.get("image", ""), "model": r.get("model", ""),
                       "preset": r.get("preset", ""), "instr": "", "variant": f"rewrite {r.get('op', '')}"}} for r in results]
