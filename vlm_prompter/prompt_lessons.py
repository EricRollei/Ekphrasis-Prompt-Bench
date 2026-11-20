"""
Prompt grades, phase 2 (spec: private_docs/SPEC_prompt_grades.md, Nov 20 2026).

1. Lessons - "what separates your great prompts from your duff ones": a language model compares samples of both and
   PROPOSES writing rules with its evidence. Nothing changes until you accept a lesson; accepted lessons are written
   to modifiers/learned_lessons.yaml as one run-wide modifier chip ("Learned lessons") you can switch on.
2. Suggested grades - from vision-check adherence (0-100) of the renders of each prompt, across checkpoints: this
   app's own checks plus, when `lora_catalog_db` is set, the LoRA Catalog's vision checks of Eric_Krea2 sweeps
   (read-only). No-LoRA reference renders are preferred, since a LoRA can pull a render off the prompt. The two
   thresholds are calibrated on the prompts you have graded, and the agreement is shown - it is a suggestion only:
   adherence measures whether renders follow a prompt, not whether the prompt is a good idea.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .outputs import prompt_sha
from .review import DB_PATH, _DB_LOCK

LESSONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modifiers",
                            "learned_lessons.yaml")

SYSTEM = (
    "You study image-generation prompts that one artist graded. GREAT prompts made images he loved; DUFF prompts "
    "failed (incoherent, collage-like, contradictory, flat, unrenderable ...). Find what the writing of GREAT prompts "
    "does that DUFF prompts do not - structure, length, staging, how subjects / light / palette / mood are described, "
    "how sources are fused, what is left out. Propose 4-8 concrete, actionable rules a prompt WRITER (another language "
    "model) could follow. Each rule must be supported by the samples; name which samples show it. No rules about "
    "subject preference or taste unless the samples are unanimous. Reply only with JSON: "
    '{"lessons": [{"rule": "...", "why": "evidence from the samples", "examples": ["G3", "D7"]}]}')


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS prompt_lessons (
        id INTEGER PRIMARY KEY AUTOINCREMENT, rule TEXT, why TEXT, examples TEXT DEFAULT '[]',
        status TEXT DEFAULT 'proposed', model TEXT, batch TEXT, created TEXT, updated TEXT)""")
    return con


# -- 1. lessons ---------------------------------------------------------------------------------------------------------
def _samples(n_each: int, seed: int) -> Tuple[List[str], List[str]]:
    with _DB_LOCK:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            rows = con.execute("SELECT prompt, grade FROM prompt_grades WHERE grade IN ('great','duff')").fetchall()
        finally:
            con.close()
    rnd = random.Random(seed)
    great = [p for p, g in rows if g == "great"]
    duff = [p for p, g in rows if g == "duff"]
    rnd.shuffle(great)
    rnd.shuffle(duff)
    clip = lambda p: p if len(p) <= 1400 else p[:1400] + " ..."
    return [clip(p) for p in great[:n_each]], [clip(p) for p in duff[:n_each]]


def propose(settings: Dict[str, Any], model: str = "__loaded__", n_each: int = 20) -> Dict[str, Any]:
    from .jobs import make_client, resolve_model
    from .vision import _parse_json
    great, duff = _samples(n_each, seed=int(_dt.datetime.now().timestamp()))
    if len(great) < 5 or len(duff) < 5:
        return {"ok": False, "error": f"grade more first: need 5+ great and 5+ duff (have {len(great)} / {len(duff)})"}
    client = make_client(settings)
    api_id, _path, _loaded = resolve_model(client, model or "__loaded__")
    text = ("GREAT prompts:\n" + "\n\n".join(f"G{i}. {p}" for i, p in enumerate(great, 1))
            + "\n\nDUFF prompts:\n" + "\n\n".join(f"D{i}. {p}" for i, p in enumerate(duff, 1)))
    r = client.chat(api_id, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                    max_tokens=4000, temperature=0.3, reasoning="off")
    out = _parse_json(r.content)
    lessons = [x for x in (out.get("lessons") or []) if isinstance(x, dict) and str(x.get("rule") or "").strip()]
    if not lessons:
        return {"ok": False, "error": "the model gave no lessons" + (" (it ran out of tokens thinking)" if r.starved else "")}
    batch = _now()
    with _DB_LOCK:
        con = _db()
        try:
            for x in lessons:
                con.execute("INSERT INTO prompt_lessons (rule, why, examples, model, batch, created, updated) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (str(x["rule"]).strip(), str(x.get("why") or "").strip(),
                             json.dumps(x.get("examples") or []), api_id, batch, batch, batch))
            con.commit()
        finally:
            con.close()
    return {"ok": True, "added": len(lessons), "model": api_id, "samples": [len(great), len(duff)]}


def lessons() -> List[Dict[str, Any]]:
    with _DB_LOCK:
        con = _db()
        try:
            rows = con.execute("SELECT id, rule, why, examples, status, model, batch FROM prompt_lessons "
                               "ORDER BY CASE status WHEN 'accepted' THEN 0 WHEN 'proposed' THEN 1 ELSE 2 END, id DESC"
                               ).fetchall()
        finally:
            con.close()
    return [{"id": r[0], "rule": r[1], "why": r[2], "examples": json.loads(r[3] or "[]"), "status": r[4],
             "model": r[5], "batch": r[6]} for r in rows]


def set_status(lesson_id: int, status: str, rule: Optional[str] = None) -> Dict[str, Any]:
    if status not in ("proposed", "accepted", "rejected"):
        raise ValueError("status must be proposed, accepted or rejected")
    with _DB_LOCK:
        con = _db()
        try:
            con.execute("UPDATE prompt_lessons SET status = ?, rule = COALESCE(?, rule), updated = ? WHERE id = ?",
                        (status, rule, _now(), int(lesson_id)))
            con.commit()
        finally:
            con.close()
    _write_modifier()
    return {"ok": True}


def _write_modifier() -> None:
    """The accepted lessons as one run-wide modifier chip (modifiers/learned_lessons.yaml)."""
    accepted = [x["rule"] for x in lessons() if x["status"] == "accepted"]
    if not accepted:
        if os.path.exists(LESSONS_FILE):
            os.remove(LESSONS_FILE)
        return
    text = "Follow these prompt-writing lessons, learned from prompts that worked and prompts that failed: " + \
        " ".join(f"({i}) {r.rstrip('.')}." for i, r in enumerate(accepted, 1))
    body = ("# Written by the Library's Lessons (accepted lessons only) - edit them there, not here.\n"
            "- id: learned_lessons\n  name: Learned lessons\n  text: " + json.dumps(text, ensure_ascii=False) + "\n")
    with open(LESSONS_FILE, "w", encoding="utf-8") as fh:
        fh.write(body)


# -- 2. suggested grades ------------------------------------------------------------------------------------------------
def _manifest(folder: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    if folder in cache:
        return cache[folder]
    man = {}
    try:
        for name in os.listdir(folder):
            if name.startswith("sweep_manifest") and name.endswith(".json"):
                with open(os.path.join(folder, name), encoding="utf-8") as fh:
                    man = json.load(fh)
                break
    except OSError:
        pass
    cells = {c.get("file"): c for c in (man.get("cells") or []) if c.get("file")}
    cache[folder] = {"cells": cells, "checkpoint": str(man.get("checkpoint") or "")}
    return cache[folder]


TEXTS: Dict[str, str] = {}          # prompt_sha -> text, filled by _adherence (for the suggestion list)


def _adherence(settings: Dict[str, Any]) -> Dict[str, List[Tuple[float, str, bool]]]:
    """prompt_sha -> [(adherence, checkpoint, is_reference)] from this app's checks and the LoRA Catalog's."""
    out: Dict[str, List[Tuple[float, str, bool]]] = defaultdict(list)
    TEXTS.clear()
    seen = set()
    cache: Dict[str, Any] = {}

    def where(folder: str, name: str) -> Tuple[str, bool, Optional[str]]:
        m = _manifest(folder, cache)
        cell = m["cells"].get(name) or {}
        over = cell.get("overrides") or {}
        return (str(over.get("checkpoint") or m["checkpoint"] or "?"), bool(over.get("lora_baseline")),
                over.get("prompt"))
    with _DB_LOCK:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            mine = con.execute("SELECT file, prompt_sha, adherence FROM checks WHERE adherence IS NOT NULL").fetchall()
        finally:
            con.close()
    for f, sha, adh in mine:
        ckpt, ref, _p = where(os.path.dirname(f), os.path.basename(f))
        out[sha].append((float(adh), ckpt, ref))
        if _p:
            TEXTS.setdefault(sha, _p)
        seen.add((os.path.basename(os.path.dirname(f)), os.path.basename(f)))
    db, root = settings.get("lora_catalog_db") or "", settings.get("sweeps_root") or ""
    if db and root and os.path.exists(db):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        try:
            rows = con.execute("SELECT sweep_id, cell_file, score FROM vision_checks WHERE score IS NOT NULL").fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
        for sid, cf, score in rows:
            if (sid, cf) in seen:
                continue
            ckpt, ref, prompt = where(os.path.join(root, sid), cf)
            if prompt:
                out[prompt_sha(prompt)].append((float(score), ckpt, ref))
                TEXTS.setdefault(prompt_sha(prompt), prompt)
    return out


def _summary(rows: List[Tuple[float, str, bool]]) -> Tuple[float, int, int, bool]:
    refs = [r for r in rows if r[2]]
    use = refs or rows
    per = defaultdict(list)
    for adh, ck, _ in use:
        per[ck].append(adh)
    means = [sum(v) / len(v) for v in per.values()]
    return round(sum(means) / len(means), 1), len(use), len(per), bool(refs)


def _grade(score: float, lo: float, hi: float) -> str:
    return "great" if score >= hi else "duff" if score < lo else "ok"


def suggestions(settings: Dict[str, Any], limit: int = 300) -> Dict[str, Any]:
    adh = _adherence(settings)
    with _DB_LOCK:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            graded = {r[0]: (r[1], r[2]) for r in con.execute("SELECT prompt_sha, grade, prompt FROM prompt_grades")}
        finally:
            con.close()
    summ = {sha: _summary(rows) for sha, rows in adh.items() if rows}
    calib = [(summ[sha][0], graded[sha][0]) for sha in summ if sha in graded]
    lo, hi, best = 60.0, 90.0, -1.0
    if len(calib) >= 10:
        for a in range(30, 95, 5):
            for b in range(a + 5, 101, 5):
                ok = sum(1 for s, g in calib if _grade(s, a, b) == g) / len(calib)
                if ok > best:
                    lo, hi, best = float(a), float(b), ok
    items = []
    for sha, (score, n, nck, ref) in summ.items():
        g = graded.get(sha)
        sug = _grade(score, lo, hi)
        text = (g[1] if g else None) or TEXTS.get(sha, "")
        items.append({"sha": sha, "prompt": text, "adherence": score, "renders": n, "checkpoints": nck, "reference_only": ref,
                      "suggested": sug, "grade": g[0] if g else None, "agrees": (g[0] == sug) if g else None})
    items.sort(key=lambda x: (x["grade"] is not None, -x["checkpoints"], -x["renders"]))
    return {"thresholds": {"duff_below": lo, "great_from": hi}, "calibrated_on": len(calib),
            "agreement": round(100 * best) if best >= 0 else None,
            "catalog_linked": bool(settings.get("lora_catalog_db") and settings.get("sweeps_root")),
            "ungraded": [x for x in items if x["grade"] is None][:limit],
            "disagree": [x for x in items if x["grade"] is not None and not x["agrees"]][:limit]}