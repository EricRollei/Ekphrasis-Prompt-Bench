"""
Prompt grades: great / ok / duff, judged on the prompt itself (spec: private_docs/SPEC_prompt_grades.md).

The grade belongs to the prompt TEXT (outputs.prompt_sha, the same key Review's ratings use), so a prompt graded once
is graded wherever it appears again - in later sweeps, lists, Photobash / Ideas / Rewrite cards.

  great  -> the prompt library (/library): search, tags, provenance, export as a list
  ok     -> kept, nothing special
  duff   -> hidden from normal use (skipped when saving lists, flagged when written again) but kept with its
            reasons and provenance in the duff archive, as the counterpoint for forensic work

LoRA judgement is NOT done here: the LoRA Catalog's Sweeps page records good / bad / unsure per LoRA cell.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

from .outputs import prompt_sha
from .review import DB_PATH, _DB_LOCK

GRADES = ("great", "ok", "duff")
REASONS = {
    "duff": ["collage / pasted together", "incoherent", "contradictory", "too long", "runaway text",
             "ignored the sources", "flat / generic", "wrong subject", "unrenderable idea"],
    "great": ["composition", "mood", "light", "surprising fusion", "clear staging", "story", "detail", "palette"],
}


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS prompt_grades (
        prompt_sha TEXT PRIMARY KEY, prompt TEXT, grade TEXT, reasons TEXT DEFAULT '[]', tags TEXT DEFAULT '[]',
        note TEXT DEFAULT '', source TEXT DEFAULT '', provenance TEXT DEFAULT '{}', reference TEXT DEFAULT '',
        created TEXT, updated TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS prompt_grades_grade ON prompt_grades(grade)")
    return con


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def set_grade(prompt: str, grade: Optional[str], reasons: Optional[List[str]] = None, tags: Optional[List[str]] = None,
              note: Optional[str] = None, source: str = "", provenance: Optional[Dict[str, Any]] = None,
              reference: str = "") -> Dict[str, Any]:
    """Grade (or with grade None, un-grade) a prompt. Reasons / tags / note left as None keep their stored values;
    provenance is merged (new keys win, old ones are kept)."""
    text = " ".join((prompt or "").split())
    if not text:
        raise ValueError("empty prompt")
    if grade is not None and grade not in GRADES:
        raise ValueError(f"grade must be one of {GRADES}")
    sha = prompt_sha(text)
    with _DB_LOCK:
        con = _db()
        try:
            if grade is None:
                con.execute("DELETE FROM prompt_grades WHERE prompt_sha = ?", (sha,))
                con.commit()
                return {"sha": sha, "grade": None}
            row = con.execute("SELECT reasons, tags, note, source, provenance, reference, created FROM prompt_grades "
                              "WHERE prompt_sha = ?", (sha,)).fetchone()
            old = {"reasons": "[]", "tags": "[]", "note": "", "source": "", "provenance": "{}", "reference": "",
                   "created": _now()}
            if row:
                old = dict(zip(("reasons", "tags", "note", "source", "provenance", "reference", "created"), row))
            prov = {**json.loads(old["provenance"] or "{}"), **{k: v for k, v in (provenance or {}).items() if v}}
            vals = (sha, text, grade,
                    json.dumps(reasons if reasons is not None else json.loads(old["reasons"] or "[]")),
                    json.dumps(tags if tags is not None else json.loads(old["tags"] or "[]")),
                    note if note is not None else (old["note"] or ""),
                    old["source"] or source, json.dumps(prov, ensure_ascii=False),
                    reference or old["reference"] or "", old["created"], _now())
            con.execute("""INSERT INTO prompt_grades (prompt_sha, prompt, grade, reasons, tags, note, source, provenance,
                           reference, created, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(prompt_sha) DO UPDATE SET prompt=excluded.prompt, grade=excluded.grade,
                           reasons=excluded.reasons, tags=excluded.tags, note=excluded.note, source=excluded.source,
                           provenance=excluded.provenance, reference=excluded.reference, updated=excluded.updated""", vals)
            con.commit()
        finally:
            con.close()
    return {"sha": sha, "grade": grade}


def grades_for(shas: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    keys = sorted({s for s in shas if s})
    out: Dict[str, Dict[str, Any]] = {}
    if not keys:
        return out
    with _DB_LOCK:
        con = _db()
        try:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT prompt_sha, grade, reasons, tags, note FROM prompt_grades WHERE prompt_sha IN ({','.join('?' * len(chunk))})"
                for sha, grade, reasons, tags, note in con.execute(q, chunk):
                    out[sha] = {"grade": grade, "reasons": json.loads(reasons or "[]"), "tags": json.loads(tags or "[]"),
                                "note": note or ""}
        finally:
            con.close()
    return out


def grade_of(text: str) -> Optional[str]:
    g = grades_for([prompt_sha(" ".join((text or "").split()))])
    return next(iter(g.values()))["grade"] if g else None


def duff_shas() -> set:
    with _DB_LOCK:
        con = _db()
        try:
            return {r[0] for r in con.execute("SELECT prompt_sha FROM prompt_grades WHERE grade = 'duff'")}
        finally:
            con.close()


def library(grade: str = "great", query: str = "", tag: str = "", limit: int = 1000) -> Dict[str, Any]:
    with _DB_LOCK:
        con = _db()
        try:
            rows = con.execute("SELECT prompt_sha, prompt, grade, reasons, tags, note, source, provenance, reference, "
                               "created, updated FROM prompt_grades ORDER BY updated DESC").fetchall()
        finally:
            con.close()
    counts = Counter(r[2] for r in rows)
    tag_counts: Counter = Counter()
    out = []
    words = [w for w in query.lower().split() if w]
    for sha, prompt, g, reasons, tags, note, source, prov, ref, created, updated in rows:
        tl = json.loads(tags or "[]")
        if g == grade:
            tag_counts.update(tl)
        if g != grade or (tag and tag not in tl) or any(w not in prompt.lower() for w in words):
            continue
        out.append({"sha": sha, "prompt": prompt, "grade": g, "reasons": json.loads(reasons or "[]"), "tags": tl,
                    "note": note or "", "source": source or "", "provenance": json.loads(prov or "{}"),
                    "reference": ref or "", "created": created, "updated": updated})
    return {"rows": out[:limit], "total": len(out), "counts": dict(counts), "tags": tag_counts.most_common(),
            "reasons": REASONS}


def _sentences(text: str) -> int:
    return max(1, len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]))


def forensics() -> Dict[str, Any]:
    """Great vs ok vs duff: counts by variant (preset version, photobash mode ...) and model, text measures, reasons."""
    with _DB_LOCK:
        con = _db()
        try:
            rows = con.execute("SELECT prompt, grade, reasons, provenance FROM prompt_grades").fetchall()
        finally:
            con.close()
    by_variant: Dict[str, Counter] = defaultdict(Counter)
    by_model: Dict[str, Counter] = defaultdict(Counter)
    measures: Dict[str, List[tuple]] = defaultdict(list)
    reasons: Dict[str, Counter] = defaultdict(Counter)
    for prompt, grade, rs, prov in rows:
        p = json.loads(prov or "{}")
        by_variant[p.get("variant") or "(unknown)"][grade] += 1
        by_model[p.get("model") or "(unknown)"][grade] += 1
        measures[grade].append((len(prompt.split()), _sentences(prompt)))
        reasons[grade].update(json.loads(rs or "[]"))

    def avg(xs, i):
        return round(sum(x[i] for x in xs) / len(xs), 1) if xs else None
    return {"by_variant": {k: dict(v) for k, v in sorted(by_variant.items())},
            "by_model": {k: dict(v) for k, v in sorted(by_model.items())},
            "measures": {g: {"n": len(m), "words": avg(m, 0), "sentences": avg(m, 1)} for g, m in measures.items()},
            "reasons": {g: c.most_common() for g, c in reasons.items()}}
