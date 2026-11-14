"""
Result cache and run log.

Cache: one JSON per key under .cache/results/<k[:2]>/<key>.json.
       key = image hash + model + instruction hash + params that change the output (vision mode, size, seed ...).
Run log: runs/<YYYY-MM-DD>/<run_id>.jsonl - one line per request (cached hits included, marked), plus a
         run header line and a run footer with the files written.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, Optional


def cache_key(**parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class ResultCache:
    def __init__(self, root: str):
        self.root = os.path.join(root, "results")

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
        os.replace(tmp, path)


class RunLog:
    def __init__(self, runs_dir: str, kind: str = "run"):
        now = _dt.datetime.now()
        self.run_id = now.strftime("%H%M%S") + f"_{kind}"
        folder = os.path.join(runs_dir, now.strftime("%Y-%m-%d"))
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, self.run_id + ".jsonl")
        n = 2
        while os.path.exists(self.path):
            self.path = os.path.join(folder, f"{self.run_id}.{n}.jsonl")
            n += 1

    def write(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", _dt.datetime.now().isoformat(timespec="seconds"))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------------------------------------------- provenance
# runs/instructions/<sha>.json   the exact system + user text for an instruction fingerprint (written once)
# runs/saved_index.jsonl         one record per prompt written to a file: file, image, prompt, model, preset, instr ...

def save_instructions(runs_dir: str, sha: str, data: Dict[str, Any]) -> None:
    folder = os.path.join(runs_dir, "instructions")
    path = os.path.join(folder, sha + ".json")
    if os.path.isfile(path):
        return
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def load_instructions(runs_dir: str, sha: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(runs_dir, "instructions", f"{sha}.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def append_index(runs_dir: str, records) -> None:
    records = list(records)
    if not records:
        return
    os.makedirs(runs_dir, exist_ok=True)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(runs_dir, "saved_index.jsonl"), "a", encoding="utf-8") as handle:
        for r in records:
            r.setdefault("saved", stamp)
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def search_index(runs_dir: str, text: str, limit: int = 20):
    """Records whose prompt equals the pasted text (whitespace-insensitive); falls back to 'contains' for fragments.
    Newest first. Each hit gets the stored instruction text when available."""
    import re as _re
    norm = _re.sub(r"\s+", " ", text).strip().lower()
    if not norm:
        return []
    path = os.path.join(runs_dir, "saved_index.jsonl")
    exact, partial = [], []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                pn = _re.sub(r"\s+", " ", str(r.get("prompt", ""))).strip().lower()
                if pn == norm:
                    exact.append(r)
                elif len(norm) >= 15 and norm in pn:
                    partial.append(r)
    except OSError:
        return []
    hits = (exact or partial)[::-1][:limit]
    for h in hits:
        h["match"] = "exact" if exact else "contains"
        if h.get("instr"):
            h["instructions"] = load_instructions(runs_dir, h["instr"])
    return hits
