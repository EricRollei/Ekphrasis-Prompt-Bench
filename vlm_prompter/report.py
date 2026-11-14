"""
Prompt Review phase 3: reports for one sweep or folder (spec docs/SPEC_review.md, section 6).

Groups the renders by version (compare-run label), checkpoint, preset and model and shows per group:
  renders, rated, mean stars, keepers %, defects % (your marks; vision suggestions separately),
  mean adherence / fidelity (your value when set, else the vision check), direction reversals and broken
  relations (detector pose check), items past the 507-token limit.
Lesson candidates: renders that follow the prompt well (adherence >= 85) but differ from the source on one aspect -
the prompt probably left that aspect unsaid. Counted per aspect, with examples; plus every note you marked as a lesson.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List

from .review import DB_PATH, _DB_LOCK, _key, load_folder, load_sweep

LESSON_ADHERENCE = 85.0
ASPECT_WORDS = {"pose": "pose / which way people face", "clothing": "clothing", "light": "light", "setting": "setting",
                "composition": "framing / camera angle", "palette": "colours", "subject": "who is shown"}


def _check_results(files: List[str]) -> Dict[str, Dict[str, Any]]:
    keys = [_key(f) for f in files]
    out: Dict[str, Dict[str, Any]] = {}
    with _DB_LOCK:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS checks (file TEXT PRIMARY KEY, source TEXT, prompt_sha TEXT, model TEXT, "
                        "checked_at TEXT, adherence REAL, fidelity REAL, defects TEXT, result TEXT, seconds REAL)")
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT file, result FROM checks WHERE file IN ({','.join('?' * len(chunk))})"
                for f, res in con.execute(q, chunk):
                    out[f] = json.loads(res)
        finally:
            con.close()
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def _pct(n, d):
    return round(100.0 * n / d) if d else None


def build(kind: str, path: str) -> Dict[str, Any]:
    data = load_sweep(path) if kind == "sweep" else load_folder(path)
    cells = data["cells"]
    checks = _check_results([c["file"] for c in cells])
    rows = []
    for c in cells:
        r = c.get("rating") or {}
        res = checks.get(_key(c["file"])) or {}
        adh, fid = res.get("adherence") or {}, res.get("fidelity") or {}
        pc = fid.get("pose_check") or {}
        problems = pc.get("problems") or []
        prov = c.get("prov") or {}
        rows.append({
            "file": c["file"], "name": c.get("name", ""), "prompt": c.get("prompt", ""),
            "version": prov.get("variant") or (prov.get("preset") or "unknown") + (" (no version label)" if prov else ""),
            "checkpoint": c.get("checkpoint") or c.get("column") or "", "preset": prov.get("preset", ""),
            "model": prov.get("model", ""), "stars": r.get("stars"), "keeper": bool(r.get("keeper")),
            "defects": r.get("defects") or [], "vdefects": (c.get("check") or {}).get("defects") or [],
            "adherence": r.get("adherence") if r.get("adherence") is not None else adh.get("score"),
            "fidelity": r.get("fidelity") if r.get("fidelity") is not None else fid.get("score"),
            "checked": bool(res), "reversal": any("faces" in p or "turned" in p or "mirror" in p for p in problems),
            "relation_break": any(p.startswith("persons ") for p in problems),
            "beyond": sum(1 for it in adh.get("items") or [] if it.get("beyond_limit")),
            "aspects": fid.get("aspects") or {}, "lesson": bool(r.get("lesson")), "note": r.get("note", ""),
        })

    def summarise(by: str) -> List[Dict[str, Any]]:
        groups = defaultdict(list)
        for row in rows:
            groups[row[by] or "-"].append(row)
        out = []
        for name, g in groups.items():
            rated = [x for x in g if x["stars"]]
            checked = [x for x in g if x["checked"]]
            with_src = [x for x in checked if x["aspects"]]
            out.append({"group": name, "renders": len(g), "rated": len(rated), "stars": _mean([x["stars"] for x in rated]),
                        "keepers": _pct(sum(x["keeper"] for x in g), len(g)),
                        "defects": _pct(sum(1 for x in g if x["defects"]), len(g)),
                        "vdefects": _pct(sum(1 for x in checked if x["vdefects"]), len(checked)),
                        "adherence": _mean([x["adherence"] for x in g]), "fidelity": _mean([x["fidelity"] for x in g]),
                        "checked": len(checked),
                        "reversals": _pct(sum(x["reversal"] for x in with_src), len(with_src)),
                        "relation_breaks": _pct(sum(x["relation_break"] for x in with_src), len(with_src)),
                        "beyond": sum(x["beyond"] for x in g)})
        return sorted(out, key=lambda x: (-(x["stars"] or 0), x["group"]))

    lessons = defaultdict(list)
    for row in rows:
        if (row["adherence"] or 0) >= LESSON_ADHERENCE:
            for asp, v in row["aspects"].items():
                if v.get("answer") == "different":
                    lessons[asp].append({"file": row["file"], "name": row["name"], "note": v.get("note", ""),
                                         "prompt": row["prompt"][:300]})
    candidates = [{"aspect": a, "label": ASPECT_WORDS.get(a, a), "count": len(v), "examples": v[:6]}
                  for a, v in sorted(lessons.items(), key=lambda kv: -len(kv[1]))]
    your_lessons = [{"file": r["file"], "name": r["name"], "note": r["note"]} for r in rows if r["lesson"] and r["note"]]
    defect_names = defaultdict(int)
    for r in rows:
        for d in r["defects"]:
            defect_names[d] += 1
    return {"source": path, "kind": kind, "renders": len(rows), "checked": sum(r["checked"] for r in rows),
            "rated": sum(1 for r in rows if r["stars"]),
            "by": {k: summarise(k) for k in ("version", "checkpoint", "preset", "model")},
            "lesson_candidates": candidates, "your_lessons": your_lessons, "defect_counts": dict(defect_names)}
