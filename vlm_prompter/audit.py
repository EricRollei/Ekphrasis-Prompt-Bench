"""
Audit: what does the vision LLM get wrong about the people that the detectors measure?

For each image we collect:
  detectors   facts.cached_analyze -> every significant person (>= MIN_PERSON_HEIGHT of the picture height),
              left to right: body / head direction, gender, age, expression, ethnicity; relations between pairs
  vlm         the LM Studio model fills in the same sheet (same size threshold, same left-to-right order)
and compare person by person (matched by order when the counts agree, else by position) plus the relations.
Disagreements go to the audit page, where Eric marks who was right. Detectors are not ground truth - verdicts are.
Face fields are compared only where the detector passed its reliability gate (facts._reliability).

Each audit run is kept as its own set ("batch": the folder name, or "provenance index").

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sqlite3
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from . import config, facts, pose_geom
from .review import DB_PATH, _DB_LOCK

FIELDS = ["people", "body", "head", "gender", "age", "expression", "ethnicity", "relation"]

VLM_SYSTEM = """You fill in a fact sheet about the people in a picture. Answer only from what is visible.
List the MAIN PEOPLE: every person whose body (head to lowest visible part) is at least about one fifth of the
picture height. List them from LEFT to RIGHT. Smaller background figures are only counted. Statues, sculptures, mannequins,
dolls and pictures of people are NOT people - leave them out.
For each person give "box": [x1, y1, x2, y2] around the whole visible person, in coordinates from 0 to 1000
relative to the picture width and height (0,0 = top-left corner).
Directions: "left" / "right" always mean the LEFT or RIGHT SIDE OF THE PICTURE the person is turned toward, as seen
by the viewer. "toward camera" = facing the viewer, "away from camera" = back to the viewer.
For each pair of main people that relate to each other, give the relation: "face each other", "turned away from each
other", "left one turned toward the right one", "right one turned toward the left one", "both face the viewer",
"both have their backs to the viewer".
Reply only with JSON:
{"people_total": <all people visible, including small ones>,
 "people": [ {"box": [x1, y1, x2, y2], "position": "left|center|right", "size": "large|medium|small",
              "body_facing": "toward camera|away from camera|three-quarter left|three-quarter right|profile left|profile right",
              "head_turned": "toward camera|left|right|away from camera",
              "gender": "female|male|unclear", "age_years": <integer>,
              "expression": "neutral|happiness|sadness|surprise|fear|disgust|anger|contempt",
              "ethnicity": "<one or two words, or unclear>"} ],
 "relations": [ {"a": <number of the left person, 1 = first in your list>, "b": <number of the right person>,
                 "relation": "<one of the relations above>"} ]}"""

TABLE = """CREATE TABLE IF NOT EXISTS audit (image TEXT PRIMARY KEY, model TEXT, checked_at TEXT, detector TEXT,
           vlm TEXT, cmp TEXT, verdicts TEXT DEFAULT '{}')"""


def _db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute(TABLE)
    if "batch" not in {r[1] for r in con.execute("PRAGMA table_info(audit)")}:
        con.execute("ALTER TABLE audit ADD COLUMN batch TEXT DEFAULT 'provenance index'")
    return con


# ----------------------------------------------------------------------------------------------- image sets
def source_images(limit: int) -> List[str]:
    """Distinct source images from the provenance index, newest first, that still exist."""
    path = os.path.join(config.RUNS_DIR, "saved_index.jsonl")
    seen, out = set(), []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    for line in reversed(lines):
        try:
            img = json.loads(line).get("image", "")
        except ValueError:
            continue
        key = os.path.normcase(img)
        if img and key not in seen and os.path.isfile(img):
            seen.add(key)
            out.append(img)
            if len(out) >= limit:
                break
    return out


def folder_images(folder: str, limit: int) -> List[str]:
    """Images in a folder (not recursive). In a sweep folder only the cell_* renders (no contact sheet)."""
    from .images import is_image, natural_key
    names = sorted((n for n in os.listdir(folder) if is_image(n)), key=natural_key)
    if any(n.startswith("sweep_manifest") and n.endswith(".json") for n in os.listdir(folder)):   # a real sweep folder
        names = [n for n in names if n.startswith("cell_")]
    return [os.path.join(folder, n) for n in names][:limit]


def batches() -> List[Dict[str, Any]]:
    with _DB_LOCK:
        con = _db()
        try:
            rows = con.execute("SELECT batch, COUNT(*), MAX(checked_at) FROM audit GROUP BY batch "
                               "ORDER BY MAX(checked_at) DESC").fetchall()
        finally:
            con.close()
    return [{"batch": b, "images": n, "last": at} for b, n, at in rows]


# ----------------------------------------------------------------------------------------------- mapping
def _vlm_body(v: str) -> str:
    v = (v or "").lower()
    if "away" in v:
        return "away"
    if "left" in v:
        return "left"
    if "right" in v:
        return "right"
    return "toward" if "toward" in v else ""


def _vlm_head(v: str) -> str:
    v = (v or "").lower()
    return "away" if "away" in v else "left" if "left" in v else "right" if "right" in v else \
        "toward" if "toward" in v else ""


def _expr(v: str) -> str:
    v = (v or "").lower().strip()
    alias = {"happy": "happiness", "sad": "sadness", "surprised": "surprise", "angry": "anger", "fearful": "fear",
             "disgusted": "disgust", "calm": "neutral", "serious": "neutral"}
    return alias.get(v, v)


def _vlm_relation(v: str) -> str:
    v = (v or "").lower()
    if "each other" in v and "away" in v:
        return "away_from_each_other"
    if "each other" in v:
        return "each_other"
    if "left one" in v and "toward" in v:
        return "a_toward_b"
    if "right one" in v and "toward" in v:
        return "b_toward_a"
    if "viewer" in v and "back" in v:
        return "both_away"
    if "viewer" in v:
        return "both_viewer"
    return ""


# vision-model words -> FairFace categories (White, Black, Latino Hispanic, East Asian, Southeast Asian, Indian,
# Middle Eastern). "asian" alone matches either Asian category.
_ETHNICITY = [
    ("Southeast Asian", ("southeast asian", "south-east asian", "filipino", "thai", "vietnamese", "indonesian", "malay")),
    ("East Asian", ("east asian", "japanese", "chinese", "korean", "taiwanese", "mongolian")),
    ("Indian", ("indian", "south asian", "pakistani", "bangladeshi", "sri lankan")),
    ("Middle Eastern", ("middle eastern", "arab", "persian", "iranian", "turkish", "north african")),
    ("Latino Hispanic", ("latino", "latina", "latinx", "hispanic", "mexican", "brazilian")),
    ("Black", ("black", "african", "afro")),
    ("White", ("white", "caucasian", "european", "nordic", "scandinavian", "slavic", "mediterranean")),
]


def _ethnicity_agree(det: str, vlm: str):
    v = (vlm or "").lower().strip()
    if not v or "unclear" in v or "unknown" in v or "mixed" in v:
        return None
    for cat, words in _ETHNICITY:
        if any(w in v for w in words):
            return cat == det
    if "asian" in v:
        return det in ("East Asian", "Southeast Asian")
    return None


def _det_body(q: Dict[str, Any]) -> str:
    f = q.get("facing") or {}
    if f.get("body") == "away from camera":
        return "away"
    if f.get("turn") in ("profile", "three-quarter") and f.get("side"):
        return f["side"]
    return "toward" if f.get("body") == "toward camera" else ""


def _det_head(q: Dict[str, Any]) -> str:
    face = q.get("face") or {}
    if face.get("usable") and face.get("yaw") is not None:
        y = face["yaw"]
        return "toward" if abs(y) < 20 else ("left" if y > 0 else "right")     # +yaw = turned to picture-left
    h = (q.get("facing") or {}).get("head", "")
    return {"toward camera": "toward", "away from camera": "away", "turned left": "left",
            "turned right": "right"}.get(h, "")


_POS_X = {"left": 0.2, "center": 0.5, "right": 0.8}


MIN_JOINTS = 9            # a detector person with fewer visible body joints is "unclear" (occluded) - not scored
MIN_IOU = 0.3


def det_box_norm(q: Dict[str, Any], size) -> List[float]:
    w, h = size
    b = q["box"]
    return [round(b[0] / w * 1000, 1), round(b[1] / h * 1000, 1), round(b[2] / w * 1000, 1), round(b[3] / h * 1000, 1)]


def _vlm_box(v: Dict[str, Any]):
    b = v.get("box")
    try:
        b = [float(x) for x in b][:4]
    except (TypeError, ValueError):
        return None
    if len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]:
        return None
    if max(b) <= 1.0:                                  # some models answer 0-1
        b = [x * 1000 for x in b]
    return [max(0.0, min(1000.0, x)) for x in b]


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(det: List[Dict[str, Any]], vlm: List[Dict[str, Any]], size) -> List[tuple]:
    """Greedy pairing by box overlap (highest IoU first); pairs below MIN_IOU are not made."""
    cands = []
    for d in det:
        db = det_box_norm(d, size)
        for v in vlm:
            vb = _vlm_box(v)
            if vb:
                cands.append((_iou(db, vb), id(d), id(v), d, v))
    cands.sort(key=lambda c: c[0], reverse=True)
    used_d, used_v, pairs = set(), set(), []
    for iou, idd, idv, d, v in cands:
        if iou < MIN_IOU or idd in used_d or idv in used_v:
            continue
        used_d.add(idd)
        used_v.add(idv)
        pairs.append((d, v, round(iou, 2)))
    return pairs


def _match(det: List[Dict[str, Any]], vlm: List[Dict[str, Any]]) -> List[tuple]:
    """(det_person, vlm_person) pairs: by order when the counts agree, else nearest position."""
    if len(det) == len(vlm):
        return list(zip(det, vlm))
    pairs, used = [], set()
    for v in vlm:
        vx = _POS_X.get(str(v.get("position", "")).lower(), 0.5)
        best = min((d for d in det if id(d) not in used), key=lambda d: abs(d["x"] - vx), default=None)
        if best is not None and abs(best["x"] - vx) <= 0.3:
            used.add(id(best))
            pairs.append((best, v))
    return pairs


def compare(det: Dict[str, Any], vlm: Dict[str, Any]) -> Dict[str, Any]:
    """{"people": {...}, "persons": [{"n", "position", fields...}], "relations": [...]} - each leaf has
    {detector, vlm, agree} with agree None when not comparable."""
    persons = det.get("persons") or []
    vpeople = [p for p in (vlm.get("people") or []) if isinstance(p, dict)]
    vn = len(vpeople)
    out: Dict[str, Any] = {"people": {"detector": len(persons), "vlm": vn, "agree": len(persons) == vn},
                           "persons": [], "relations": []}
    vindex = {id(p): i + 1 for i, p in enumerate(vpeople)}
    det_by_vn = {}
    size = det.get("size") or [1000, 1000]
    use_boxes = any(_vlm_box(v) for v in vpeople)
    pairs = _match_boxes(persons, vpeople, size) if use_boxes else [(d, v, None) for d, v in _match(persons, vpeople)]
    paired_d, paired_v = {id(p[0]) for p in pairs}, {id(p[1]) for p in pairs}
    out["boxes"] = {"det": [{"n": d["n"], "box": det_box_norm(d, size),
                             "unclear": int(sum(1 for s in d["scores"][:17] if s >= pose_geom.MIN_SCORE)) < MIN_JOINTS}
                            for d in persons],
                    "vlm": [{"i": vindex[id(v)], "box": _vlm_box(v)} for v in vpeople if _vlm_box(v)]}
    out["only_detector"] = [d["n"] for d in persons if id(d) not in paired_d]
    out["only_model"] = [vindex[id(v)] for v in vpeople if id(v) not in paired_v]
    out["matched_by"] = "boxes" if use_boxes else "order"
    unclear = {b["n"] for b in out["boxes"]["det"] if b["unclear"]}
    for d, v, iou in pairs:
        det_by_vn[vindex[id(v)]] = d["n"]
        for b in out["boxes"]["vlm"]:
            if b["i"] == vindex[id(v)]:
                b["n"] = d["n"]
        if d["n"] in unclear:                          # occluded: counted, never scored
            continue
        face = d.get("face") or {}
        row: Dict[str, Any] = {"n": d["n"], "position": d["position"], "iou": iou}
        db, vb = _det_body(d), _vlm_body(v.get("body_facing", ""))
        row["body"] = {"detector": db, "vlm": vb, "agree": pose_geom.directions_agree(db, vb) if db and vb else None}
        dh, vh = _det_head(d), _vlm_head(v.get("head_turned", ""))
        row["head"] = {"detector": dh, "vlm": vh, "agree": pose_geom.directions_agree(dh, vh) if dh and vh else None}
        if face.get("gender_ok"):
            vg = str(v.get("gender") or "").lower()
            row["gender"] = {"detector": face["gender"], "vlm": vg, "agree": None if vg in ("", "unclear") else vg == face["gender"]}
        if face.get("age_ok"):
            try:
                va = int(v.get("age_years"))
            except (TypeError, ValueError):
                va = None
            row["age"] = {"detector": face["age"], "vlm": va, "agree": None if va is None else abs(va - face["age"]) <= 7,
                          "diff": None if va is None else va - face["age"]}
        if face.get("expression_ok"):
            ve = _expr(v.get("expression", ""))
            row["expression"] = {"detector": face["expression"], "vlm": ve, "agree": (ve == face["expression"]) if ve else None}
        if face.get("ethnicity_ok"):
            row["ethnicity"] = {"detector": face.get("ethnicity", ""), "vlm": v.get("ethnicity", ""),
                                "agree": _ethnicity_agree(face.get("ethnicity", ""), v.get("ethnicity", ""))}
        out["persons"].append(row)
    drel = {(r["a"], r["b"]): r["class"] for r in det.get("relations") or []}
    for r in vlm.get("relations") or []:
        try:
            a, b = det_by_vn.get(int(r.get("a"))), det_by_vn.get(int(r.get("b")))
        except (TypeError, ValueError):
            continue
        if not a or not b:
            continue
        a, b = min(a, b), max(a, b)
        vc, dc = _vlm_relation(r.get("relation", "")), drel.get((a, b), "")
        out["relations"].append({"a": a, "b": b, "detector": pose_geom.RELATION_TEXT.get(dc, "no clear relation"),
                                 "vlm": pose_geom.RELATION_TEXT.get(vc, str(r.get("relation", ""))),
                                 "agree": pose_geom.relations_agree(dc, vc) if vc and dc else None})
    return out


def _leaves(cmp: Dict[str, Any]):
    """(key, field, label, leaf) for every comparison in a cmp - also reads the old one-person format."""
    if "persons" not in cmp:                                       # first audit format (main person only)
        for f, c in cmp.items():
            if isinstance(c, dict) and "agree" in c:
                yield f, f, f, c
        return
    yield "people", "people", "main people", cmp["people"]
    for p in cmp["persons"]:
        for f in ("body", "head", "gender", "age", "expression", "ethnicity"):
            if f in p:
                yield f"p{p['n']}.{f}", f, f"{f} - person {p['n']} ({p['position']})", p[f]
    for r in cmp["relations"]:
        yield f"r{r['a']}-{r['b']}", "relation", f"relation - persons {r['a']} and {r['b']}", r


# ----------------------------------------------------------------------------------------------- results
def results(batch: Optional[str] = None) -> Dict[str, Any]:
    all_b = batches()
    batch = batch or (all_b[0]["batch"] if all_b else "")
    with _DB_LOCK:
        con = _db()
        try:
            rows = con.execute("SELECT image, model, checked_at, detector, vlm, cmp, verdicts FROM audit WHERE batch = ? "
                               "ORDER BY checked_at", (batch,)).fetchall()
        finally:
            con.close()
    summary = {f: {"compared": 0, "agree": 0, "detector_right": 0, "vlm_right": 0, "both_wrong": 0, "judged": 0}
               for f in FIELDS}
    items = []
    for image, model, at, det, vlm, cmp, ver in rows:
        cmp, ver = json.loads(cmp), json.loads(ver or "{}")
        leaves = []
        for key, field, label, c in _leaves(cmp):
            s = summary[field]
            if c.get("agree") is not None:
                s["compared"] += 1
                s["agree"] += 1 if c["agree"] else 0
            v = ver.get(key)
            if v in ("detector", "vlm", "both", "unclear"):
                s["judged"] += 1
                if v != "unclear":
                    s[{"detector": "detector_right", "vlm": "vlm_right", "both": "both_wrong"}[v]] += 1
            leaves.append({"key": key, "field": field, "label": label, "detector": c.get("detector"),
                           "vlm": c.get("vlm"), "agree": c.get("agree"), "diff": c.get("diff")})
        items.append({"image": image, "model": model, "checked_at": at, "leaves": leaves, "verdicts": ver,
                      "boxes": cmp.get("boxes"), "only_detector": cmp.get("only_detector", []),
                      "only_model": cmp.get("only_model", []), "matched_by": cmp.get("matched_by", ""),
                      "facts_text": facts.facts_text(json.loads(det), include_ethnicity=True)})
    return {"items": items, "summary": summary, "batch": batch, "batches": all_b}


def set_verdict(image: str, key: str, verdict: str) -> None:
    if not re.fullmatch(r"(people|body|head|gender|age|expression|ethnicity|relation|p\d+\.\w+|r\d+-\d+)", key or "") \
            or verdict not in ("detector", "vlm", "both", "unclear", ""):
        raise ValueError("bad field or verdict")
    with _DB_LOCK:
        con = _db()
        try:
            row = con.execute("SELECT verdicts FROM audit WHERE image = ?", (image,)).fetchone()
            if not row:
                raise KeyError(image)
            v = json.loads(row[0] or "{}")
            if verdict:
                v[key] = verdict
            else:
                v.pop(key, None)
            con.execute("UPDATE audit SET verdicts = ? WHERE image = ?", (json.dumps(v), image))
            con.commit()
        finally:
            con.close()


# ----------------------------------------------------------------------------------------------- job
class AuditJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {"running": False}
        self.stop_flag = threading.Event()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def start(self, settings: Dict[str, Any], limit: int, model: str, redo: bool = False,
              folder: str = "") -> Dict[str, Any]:
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("an audit is already running")
            self.state = {"running": True, "done": 0, "total": 0, "errors": 0, "last_error": "", "started": time.time()}
            self.stop_flag = threading.Event()
        threading.Thread(target=self._run, args=(settings, limit, model, redo, folder), daemon=True).start()
        return self.status()

    def stop(self) -> None:
        self.stop_flag.set()

    def _run(self, settings, limit, model, redo, folder="") -> None:
        from .jobs import make_client, resolve_model
        from .model_caps import caps_for, normalise_mode
        from .vision import Checker, _image_url
        try:
            if folder:
                images = folder_images(folder, limit)
                batch = os.path.basename(folder.rstrip("\\/")) or folder
            else:
                images = source_images(limit)
                batch = "provenance index"
            with self.lock:
                self.state["batch"] = batch
            if not redo:
                with _DB_LOCK:
                    con = _db()
                    try:
                        have = {r[0] for r in con.execute("SELECT image FROM audit")}
                    finally:
                        con.close()
                images = [i for i in images if os.path.normcase(i) not in have]
            client = make_client(settings)
            api_id, gguf, _ = resolve_model(client, model or "__loaded__")
            caps = caps_for(gguf)
            checker = Checker(client, api_id, normalise_mode("off", caps), caps)
            with self.lock:
                self.state.update(total=len(images), model=api_id)
            for n, img in enumerate(images, 1):
                if self.stop_flag.is_set():
                    break
                with self.lock:
                    self.state["current"] = os.path.basename(img)
                try:
                    det = facts.cached_analyze(img)
                    vlm = checker._ask(VLM_SYSTEM, [{"type": "text", "text": "Fill in the fact sheet for this picture."},
                                                    {"type": "image_url", "image_url": {"url": _image_url(img)}}], 1200)
                    cmp = compare(det, vlm)
                    with _DB_LOCK:
                        con = _db()
                        try:
                            con.execute("INSERT OR REPLACE INTO audit (image, model, checked_at, detector, vlm, cmp, verdicts, batch) "
                                        "VALUES (?, ?, ?, ?, ?, ?, '{}', ?)",
                                        (os.path.normcase(img), api_id, _dt.datetime.now().isoformat(timespec="seconds"),
                                         json.dumps(det), json.dumps(vlm), json.dumps(cmp), batch))
                            con.commit()
                        finally:
                            con.close()
                except Exception as exc:
                    with self.lock:
                        self.state["errors"] += 1
                        self.state["last_error"] = f"{os.path.basename(img)}: {type(exc).__name__}: {exc}"
                with self.lock:
                    self.state["done"] = n
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.state.update(running=False, current="", finished=time.time())


JOB = AuditJob()
