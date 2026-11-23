"""
Prompt Review phase 2: vision checks of renders (spec docs/SPEC_review.md v0.2, section 5).

Adapted from the LoRA Catalog's lora_catalog/vision_check.py (checklist-per-prompt idea, prompts and the adherence
formula: core x2 / detail x1, yes / partly / no, an extra or substituted main subject caps the score at 60).
Changes for prompt review:
  * checklist items carry a KIND (pose, orientation, clothing ...) and the exact PHRASE they come from, so we can see
    which kinds of instruction Krea 2 drops and whether an item sat beyond the 507-token encoder limit
  * FIDELITY: a second look with the source image and the render side by side, plus "were the requested changes made"
  * defect SUGGESTIONS (signature, text, nudity, anatomy, physics, main subject) are stored apart from your own defect
    marks - you accept or ignore them in the Review tab; nothing you set is overwritten

Uses the LM Studio model chosen in the header (default: whatever is loaded), thinking off unless chosen.
Results: review.sqlite tables checklists + checks.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import io
import json
import os
import re
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps

from . import config
from .review import DB_PATH, _DB_LOCK, _key, load_folder, load_sweep

CHECKLIST_VERSION = "pr1"
TOKEN_LIMIT = 507                 # Krea 2: 512 minus template suffix (Eric_Krea2 _prompt_budget.py)
EXTRA_CAP = 60.0
MINOR_EXTRA = 5.0
KINDS = ("subject", "count", "attribute", "clothing", "pose", "orientation", "hands", "spatial", "setting",
         "text", "lighting", "camera", "style")

CHECKLIST_SYSTEM = """You turn an image-generation prompt into a checklist for judging a picture made from it.
Write 5 to 10 requirements a person could verify by looking at the picture:
- each main subject; give a count only for PEOPLE and the central subject ("exactly one woman"), never for props;
- the key visible attributes of each subject: hair, skin, clothing, colours and materials;
- pose, and WHICH WAY each person faces (toward / away from the camera, turned left / right) when the prompt says;
- what the hands do and what the body touches, when stated;
- spatial relations (left of, behind, in front of), the setting, the medium if stated;
- the lighting idea only if it is concrete and visible (time of day, light coming from one side).
One fact per item, plain words. Leave out: artist names, quality words, moods that cannot be seen, anatomy, and
anything the prompt does not ask for.
For each item give: "weight": "core" (the picture fails without it: main subjects, counts, key attributes) or "detail";
"kind": one of subject, count, attribute, clothing, pose, orientation, hands, spatial, setting, text, lighting, camera,
style; "phrase": the shortest exact words copied from the prompt that this item comes from.
Also say whether the prompt asks for visible text / lettering, and whether it asks for nudity.
Reply only with JSON: {"items": [{"requirement": "...", "weight": "core|detail", "kind": "...", "phrase": "..."}],
"asks_for_text": true|false, "asks_for_nudity": true|false}"""

IMAGE_SYSTEM = """You inspect an AI-generated image. Answer only from what is visible. Look at the whole image,
including corners and edges.
1. signature: is there a signature, artist's initials, monogram, logo or watermark anywhere?
2. text: is there any text, letters or writing anywhere? If yes, copy up to 8 words of it.
3. nudity: is there nudity (bare breasts, genitals, bare buttocks) or sexual content?
4. anatomy: is there a clear anatomical error - extra or missing fingers, limbs or heads, merged bodies, a joint bent
   the wrong way, a face badly deformed? Minor softness is not an error.
5. physics: is anything physically impossible - a person or object floating without support, a pose that could not
   hold its weight, bodies or objects passing through each other, water or light behaving impossibly?
6. For each numbered requirement: "yes" = clearly there and right, "partly" = there but wrong in some way (colour,
   number, attribute, direction) or hard to make out, "no" = missing or wrong. Add a few words for partly / no.
7. extra_subjects: compare with the FULL PROMPT: "main" = a PERSON or the central subject the prompt does not ask for,
   a different number of PEOPLE than asked, or a main subject replaced by something else; "minor" = an extra or
   duplicated prop or background thing; "none" = nothing like that. Nudity or clothing is never an extra subject.
Reply only with JSON: {"signature": "yes|no", "text": "yes|no", "text_reads": "", "nudity": "yes|no",
"anatomy": "yes|no", "anatomy_what": "", "physics": "yes|no", "physics_what": "",
"answers": [{"n": 1, "answer": "yes|partly|no", "note": ""}], "extra_subjects": "none|minor|main", "extra_what": ""}"""

FIDELITY_SYSTEM = """You compare two images. IMAGE 1 is the SOURCE photograph. IMAGE 2 is a NEW picture generated
from a text description of the source. Judge how closely IMAGE 2 recreates IMAGE 1, taking the REQUESTED CHANGES into
account: a requested change is supposed to differ and must not count against the match.
For each aspect answer "same", "close" or "different", with a few words when not "same":
subject (who / what is shown, how many people, their look), clothing, pose (body position, which way each person faces,
what the hands do and touch), setting, composition (framing, camera angle, where things sit in the frame),
light (direction, time of day, quality), palette (main colours).
Separately, for the MAIN PERSON in each image (the most prominent one; "none" if there is no person), say which way
the BODY faces and which way the HEAD / FACE points, each as one of: "toward camera", "away from camera",
"three-quarter left", "three-quarter right", "profile left", "profile right" (left / right as seen in the picture).
Then for each REQUESTED CHANGE say whether IMAGE 2 applied it: "yes", "no", or "n/a" when the change is only about
how the description was worded and cannot be seen in a picture.
Reply only with JSON: {"aspects": {"subject": {"answer": "same|close|different", "note": ""}, "clothing": {...},
"pose": {...}, "setting": {...}, "composition": {...}, "light": {...}, "palette": {...}},
"facing": {"image1_body": "", "image1_head": "", "image2_body": "", "image2_head": ""},
"changes": [{"n": 1, "applied": "yes|no|n/a", "note": ""}]}"""

FIDELITY_WEIGHTS = {"subject": 2, "clothing": 2, "pose": 2, "setting": 1, "composition": 1, "light": 1, "palette": 1}
FIDELITY_POINTS = {"same": 1.0, "close": 0.6, "different": 0.0}
CHANGE_MISS = 15.0                # each requested change NOT applied costs this many points

TABLES = """
CREATE TABLE IF NOT EXISTS checklists (key TEXT PRIMARY KEY, prompt TEXT, model TEXT, data TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS checks (file TEXT PRIMARY KEY, source TEXT, prompt_sha TEXT, model TEXT, checked_at TEXT,
    adherence REAL, fidelity REAL, defects TEXT, result TEXT, seconds REAL);
CREATE INDEX IF NOT EXISTS checks_source ON checks(source);
"""


# ===================================================================================================== helpers
def _db():
    import sqlite3
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.executescript(TABLES)
    return con


def _yes(v: Any) -> bool:
    return v is True or str(v or "").strip().lower().startswith("y")


def _parse_json(text: str) -> Dict[str, Any]:
    t = re.sub(r"```[a-zA-Z]*", "", text or "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON in the model's answer")
    return json.loads(t[a:b + 1])


def _image_url(path: str, side: int = 1024) -> str:
    from .images import contact_sheet, is_video
    if is_video(path):                               # a video render / source is judged from its contact sheet
        im = contact_sheet(path)
        im.thumbnail((side * 2, side * 2), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def approx_tokens(text: str) -> int:
    """Rough Qwen-tokenizer count (~1.3 tokens per word, punctuation adds a little). Labelled approximate in the UI."""
    words = len(re.findall(r"\w+", text))
    punct = len(re.findall(r"[^\w\s]", text))
    return int(words * 1.3 + punct * 0.5)


def phrase_position(prompt: str, phrase: str) -> Optional[int]:
    """Approximate token position where the phrase starts in the prompt (None when not found)."""
    if not phrase:
        return None
    i = prompt.lower().find(phrase.lower().strip(" .,"))
    if i < 0:
        words = [w for w in re.findall(r"\w+", phrase.lower()) if len(w) > 3][:3]
        if not words:
            return None
        m = re.search(r"\b" + r"\W+(?:\w+\W+){0,3}".join(map(re.escape, words)), prompt.lower())
        if not m:
            return None
        i = m.start()
    return approx_tokens(prompt[:i])


def _side(direction: str) -> str:
    """Coarse facing class: toward / away / left / right (three-quarter and profile of one side count as that side)."""
    d = direction.lower()
    if "away" in d or "back" in d:
        return "away"
    if "left" in d:
        return "left"
    if "right" in d:
        return "right"
    if "toward" in d or "camera" in d or "front" in d:
        return "toward"
    return d


def requested_changes(prov: Optional[Dict[str, Any]]) -> List[str]:
    """The changes the prompt was asked to make relative to the source (modifiers + run note), as plain text."""
    if not prov:
        return []
    from .instructions import load_modifiers
    mods = load_modifiers(config.MODIFIERS_DIR)
    out = []
    for m in prov.get("modifiers") or []:
        mid, _, value = m.partition("=")
        if mid == "extra_detail":                     # wording-only modifier, nothing visible to check
            continue
        text = mods[mid].text.replace("{value}", value) if mid in mods else m
        out.append(text)
    # free-text run notes are left out: they are usually about wording ("omit camera words"), not the picture
    return out


# ===================================================================================================== checker
class Checker:
    def __init__(self, client, model: str, mode: str, caps: Dict[str, bool]):
        self.client, self.model, self.mode, self.caps = client, model, mode, caps

    def _ask(self, system: str, user_content, max_tokens: int = 1200) -> Dict[str, Any]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        r = self.client.chat(self.model, msgs, max_tokens=max_tokens + (4096 if self.mode not in ("off", "") and
                                                                          not self.mode.startswith("i") else 0),
                             temperature=0.0, reasoning=self.mode, caps=self.caps)
        return _parse_json(r.content)

    def checklist(self, prompt: str) -> Dict[str, Any]:
        key = hashlib.sha1((CHECKLIST_VERSION + "|" + prompt.strip().lower()).encode("utf-8")).hexdigest()[:24]
        with _DB_LOCK:
            con = _db()
            try:
                row = con.execute("SELECT data FROM checklists WHERE key = ?", (key,)).fetchone()
            finally:
                con.close()
        if row:
            return json.loads(row[0])
        out = self._ask(CHECKLIST_SYSTEM, prompt, 1400)
        items = []
        for it in (out.get("items") or [])[:10]:
            if not isinstance(it, dict) or not it.get("requirement"):
                continue
            kind = str(it.get("kind", "")).lower()
            pos = phrase_position(prompt, str(it.get("phrase", "")))
            items.append({"requirement": it["requirement"], "weight": "core" if it.get("weight") == "core" else "detail",
                          "kind": kind if kind in KINDS else "other", "phrase": str(it.get("phrase", ""))[:200],
                          "token_pos": pos, "beyond_limit": pos is not None and pos >= TOKEN_LIMIT})
        data = {"items": items, "asks_for_text": bool(out.get("asks_for_text")),
                "asks_for_nudity": bool(out.get("asks_for_nudity")), "prompt_tokens": approx_tokens(prompt)}
        with _DB_LOCK:
            con = _db()
            try:
                con.execute("INSERT OR REPLACE INTO checklists VALUES (?, ?, ?, ?, ?)",
                            (key, prompt, self.model, json.dumps(data, ensure_ascii=False),
                             _dt.datetime.now().isoformat(timespec="seconds")))
                con.commit()
            finally:
                con.close()
        return data

    def adherence(self, cell: Dict[str, Any], cl: Dict[str, Any]) -> Dict[str, Any]:
        prompt = cell.get("prompt") or ""
        req = "\n".join(f"{n}. {i['requirement']}" for n, i in enumerate(cl["items"], 1)) or "(no requirements)"
        text = "FULL PROMPT:\n" + (prompt or "(none)") + "\n\nRequirements:\n" + req
        out = self._ask(IMAGE_SYSTEM, [{"type": "text", "text": text},
                                       {"type": "image_url", "image_url": {"url": _image_url(cell["file"])}}], 1400)
        defects, reasons = [], []
        if _yes(out.get("signature")):
            defects.append("signature")
        if _yes(out.get("text")) and not cl.get("asks_for_text"):
            defects.append("text")
        if _yes(out.get("nudity")) and not cl.get("asks_for_nudity"):
            defects.append("nudity")
        # anatomy / physics count only when the model says WHAT is wrong - a bare "yes" from a small model is noise
        for name in ("anatomy", "physics"):
            what = str(out.get(name + "_what") or "").strip()
            if _yes(out.get(name)) and len(what) > 3:
                defects.append(name)
                reasons.append(f"{name}: {what}")
        answers = {}
        for a in out.get("answers") or []:
            try:
                answers[int(a.get("n", 0))] = a
            except (TypeError, ValueError, AttributeError):
                continue
        items, pts, wsum = [], 0.0, 0.0
        for n, item in enumerate(cl["items"], 1):
            w = 2.0 if item["weight"] == "core" else 1.0
            a = answers.get(n) or {}
            ans = str(a.get("answer", "no")).lower()
            p = 1.0 if ans.startswith("y") else 0.5 if ans.startswith("p") else 0.0
            pts, wsum = pts + w * p, wsum + w
            items.append({**item, "answer": "yes" if p == 1 else "partly" if p == 0.5 else "no",
                          "note": str(a.get("note") or "")})
        score = round(100.0 * pts / wsum, 1) if wsum else None
        sev = str(out.get("extra_subjects") or "none").strip().lower()
        what = str(out.get("extra_what") or "").strip()
        if sev.startswith("main"):
            defects.append("subject")
            reasons.append("extra or substituted main subject: " + (what or "?"))
            score = min(score, EXTRA_CAP) if score is not None else None
        elif sev.startswith("minor") and score is not None:
            reasons.append("minor extra: " + (what or "?"))
            score = max(0.0, score - MINOR_EXTRA)
        return {"score": score, "items": items, "defects": defects, "reasons": reasons,
                "text_reads": str(out.get("text_reads") or "")}

    def fidelity(self, cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prov = cell.get("prov") or {}
        src = prov.get("source_image")
        if not src or not os.path.isfile(src):
            return None
        # Nov 20 2026: a photobash prompt mixes several images - comparing the render with ONE of them says nothing,
        # so there is no fidelity score for it (adherence to the prompt still applies)
        if prov.get("mode") == "photobash" or len(prov.get("sources") or []) > 1:
            return None
        changes = requested_changes(prov)
        text = "REQUESTED CHANGES:\n" + ("\n".join(f"{n}. {c}" for n, c in enumerate(changes, 1)) or "(none)")
        out = self._ask(FIDELITY_SYSTEM, [{"type": "text", "text": text + "\n\nIMAGE 1 (source):"},
                                          {"type": "image_url", "image_url": {"url": _image_url(src)}},
                                          {"type": "text", "text": "IMAGE 2 (generated):"},
                                          {"type": "image_url", "image_url": {"url": _image_url(cell["file"])}}], 1200)
        aspects, pts, wsum = {}, 0.0, 0.0
        for name, w in FIDELITY_WEIGHTS.items():
            a = (out.get("aspects") or {}).get(name) or {}
            ans = str(a.get("answer", "different")).lower()
            ans = "same" if ans.startswith("s") else "close" if ans.startswith("c") else "different"
            aspects[name] = {"answer": ans, "note": str(a.get("note") or "")}
            pts, wsum = pts + w * FIDELITY_POINTS[ans], wsum + w
        # detector pose check (Nov 13): the vision model misses facing reversals; keypoints + head pose do not
        pose_chk = None
        try:
            from .facts import pose_check
            pose_chk = pose_check(src, cell["file"])
        except Exception as exc:                       # detectors optional: never fail the check over them
            pose_chk = {"error": f"{type(exc).__name__}: {exc}"}
        if pose_chk and pose_chk.get("problems"):
            old = aspects["pose"]["answer"]
            aspects["pose"] = {"answer": "different", "note": "detectors: " + "; ".join(pose_chk["problems"])}
            pts -= FIDELITY_WEIGHTS["pose"] * FIDELITY_POINTS[old]
        if pose_chk and (pose_chk.get("drift") or {}).get("gender") and aspects["subject"]["answer"] != "different":
            old = aspects["subject"]["answer"]
            aspects["subject"] = {"answer": "different", "note": "detectors: gender " + pose_chk["drift"]["gender"]}
            pts -= FIDELITY_WEIGHTS["subject"] * (FIDELITY_POINTS[old] - FIDELITY_POINTS["different"])
        # facing: compared here, not judged by the model (it calls reversed poses "same")
        f = out.get("facing") or {}
        facing = {k: str(f.get(k) or "").strip().lower() for k in ("image1_body", "image1_head", "image2_body", "image2_head")}
        mismatch = [] if (pose_chk and pose_chk.get("problems")) else [part for part in ("body", "head")
                    if facing[f"image1_{part}"] and facing[f"image2_{part}"] and "none" not in (facing[f"image1_{part}"], facing[f"image2_{part}"])
                    and _side(facing[f"image1_{part}"]) != _side(facing[f"image2_{part}"])]
        if mismatch:
            old = aspects["pose"]["answer"]
            note = "; ".join(f"{p} faces {facing['image1_' + p]} in the source, {facing['image2_' + p]} in the render" for p in mismatch)
            aspects["pose"] = {"answer": "different", "note": note}
            pts -= FIDELITY_WEIGHTS["pose"] * FIDELITY_POINTS[old]
        score = 100.0 * pts / wsum
        applied = []
        got = {}
        for c in out.get("changes") or []:
            try:
                got[int(c.get("n", 0))] = c
            except (TypeError, ValueError, AttributeError):
                continue
        for n, ch in enumerate(changes, 1):
            raw = str((got.get(n) or {}).get("applied", "yes")).strip().lower()
            if raw.startswith("n/a") or raw in ("na", "none", "not applicable"):
                continue                                   # wording-only instruction: nothing visible to judge
            ok = _yes(raw)
            applied.append({"change": ch, "applied": ok, "note": str((got.get(n) or {}).get("note") or "")})
            if not ok:
                score -= CHANGE_MISS
        return {"score": round(max(0.0, score), 1), "aspects": aspects, "changes": applied, "source": src,
                "facing": facing, "facing_mismatch": mismatch, "pose_check": pose_chk}

    def check(self, cell: Dict[str, Any], do_fidelity: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt = cell.get("prompt") or ""
        cl = self.checklist(prompt) if prompt else {"items": [], "asks_for_text": False, "asks_for_nudity": False,
                                                    "prompt_tokens": 0}
        adh = self.adherence(cell, cl)
        fid = self.fidelity(cell) if do_fidelity else None
        try:                                          # Nov 20: judged against where the prompt came from
            org = origin_check(self, cell) if do_fidelity else None
        except Exception as exc:
            org = {"error": f"{type(exc).__name__}: {exc}"}
        return {"adherence": adh, "fidelity": fid, "origin": org, "checklist": cl, "model": self.model, "mode": self.mode,
                "seconds": round(time.perf_counter() - t0, 1)}


# ===================================================================================================== storage
def save_check(cell: Dict[str, Any], source: str, res: Dict[str, Any]) -> None:
    with _DB_LOCK:
        con = _db()
        try:
            con.execute("INSERT OR REPLACE INTO checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                _key(cell["file"]), os.path.normcase(source), cell.get("sha", ""), res["model"],
                _dt.datetime.now().isoformat(timespec="seconds"), res["adherence"]["score"],
                (res["fidelity"] or {}).get("score"), json.dumps(res["adherence"]["defects"]),
                json.dumps(res, ensure_ascii=False), res["seconds"]))
            con.commit()
        finally:
            con.close()


def checked_files(files: List[str]) -> Dict[str, Dict[str, Any]]:
    """{normcased path: {adherence, fidelity, defects}} for the files already checked."""
    keys = [_key(f) for f in files]
    out: Dict[str, Dict[str, Any]] = {}
    if not keys:
        return out
    with _DB_LOCK:
        con = _db()
        try:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT file, adherence, fidelity, defects FROM checks WHERE file IN ({','.join('?' * len(chunk))})"
                for row in con.execute(q, chunk):
                    out[row[0]] = {"adherence": row[1], "fidelity": row[2], "defects": json.loads(row[3] or "[]")}
        finally:
            con.close()
    return out


def attach_checks(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    found = checked_files([c["file"] for c in cells])
    for c in cells:
        c["check"] = found.get(_key(c["file"]))
    return cells


def check_detail(file: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        con = _db()
        try:
            row = con.execute("SELECT model, checked_at, result FROM checks WHERE file = ?", (_key(file),)).fetchone()
        finally:
            con.close()
    if not row:
        return None
    res = json.loads(row[2])
    res["checked_at"] = row[1]
    return res


# ===================================================================================================== background job
class CheckJob:
    """One check run at a time (it shares LM Studio with generation runs)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {"running": False}
        self.stop_flag = threading.Event()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def start(self, settings: Dict[str, Any], kind: str, path: str, model: str, mode: str, force: bool,
              do_fidelity: bool, only: Optional[List[str]] = None) -> Dict[str, Any]:
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("a check run is already going")
            self.state = {"running": True, "source": path, "done": 0, "total": 0, "errors": 0, "last_error": "",
                          "current": "", "started": time.time(), "model": ""}
            self.stop_flag = threading.Event()
        threading.Thread(target=self._run, args=(settings, kind, path, model, mode, force, do_fidelity, only),
                         daemon=True).start()
        return self.status()

    def stop(self) -> None:
        self.stop_flag.set()

    def _set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def _run(self, settings, kind, path, model, mode, force, do_fidelity, only) -> None:
        from .jobs import make_client, resolve_model
        from .model_caps import caps_for, normalise_mode
        try:
            data = load_sweep(path) if kind == "sweep" else load_folder(path)
            cells = [c for c in data["cells"] if c.get("prompt")]
            if only is not None:                           # an explicit list, even empty, limits the run
                want = {_key(f) for f in only}
                cells = [c for c in cells if _key(c["file"]) in want]
            if not force:
                done = checked_files([c["file"] for c in cells])
                cells = [c for c in cells if _key(c["file"]) not in done]
            client = make_client(settings)
            api_id, gguf, _ = resolve_model(client, model or "__loaded__")
            caps = caps_for(gguf)
            checker = Checker(client, api_id, normalise_mode(mode or "off", caps), caps)
            self._set(total=len(cells), model=api_id)
            for i, cell in enumerate(cells, 1):
                if self.stop_flag.is_set():
                    break
                self._set(current=os.path.basename(cell["file"]))
                try:
                    save_check(cell, path, checker.check(cell, do_fidelity))
                except Exception as exc:                 # one bad cell never stops the run
                    with self.lock:
                        self.state["errors"] += 1
                        self.state["last_error"] = f"{os.path.basename(cell['file'])}: {type(exc).__name__}: {exc}"
                self._set(done=i)
        except Exception as exc:
            traceback.print_exc()
            self._set(last_error=f"{type(exc).__name__}: {exc}")
        finally:
            self._set(running=False, current="", finished=time.time())


JOB = CheckJob()


# ===================================================================================================== origin checks (Nov 20)
# Judge a render against where its prompt came from: photobash sources, the rewrite's original prompt, the idea.
ORIGIN_PHOTOBASH = """You judge one PHOTOBASH. A prompt writer looked at several SOURCE pictures and wrote ONE prompt for a new
picture that fuses them; an image model then rendered that prompt. You get the sources (each with what it was meant to
contribute), the PROMPT, and the RENDER. Answer only from what you see and read. The point is to find WHERE a weak
result comes from: the prompt, the combination of sources, or the image model.

Return only JSON:
{"sources": [{"source": 1, "in_prompt": "yes|partly|no", "note": "what of it reached the prompt, or what was lost"}],
 "prompt_fusion": 0-100,      // does the PROMPT reconcile the sources into ONE believable scene - one place, one light,
                              // one scale, one moment, consistent medium - instead of listing incompatible things?
 "prompt_issues": ["..."],    // contradictions, collage wording, things an image cannot show ...
 "source_clash": 0-100,       // how hard are THESE sources to combine at all? 0 = natural fit, 100 = nearly impossible
                              // (clashing media, scales, viewpoints, eras, lighting)
 "clash_notes": "...",
 "render_follows_prompt": 0-100,
 "render_coherence": 0-100,   // is the RENDER one coherent picture - no collage, double exposure, broken anatomy or physics?
 "likely_fault": "prompt|combination|image model|none",
 "why": "one or two sentences"}"""

ORIGIN_REWRITE = """You judge a prompt REWRITE and the picture made from the new prompt. You get the ORIGINAL prompt (and the
image it described, when known), the operation and the instruction that was asked for, the NEW prompt and the RENDER.
Answer only from what you see and read.

Return only JSON:
{"change_made": 0-100,          // did the NEW prompt carry out the instruction everywhere it applies?
 "kept": 0-100,                 // did it keep what should stay (subject, setting, details the instruction did not touch)?
 "render_shows_change": 0-100,  // does the RENDER show the requested change?
 "issues": ["..."],
 "likely_fault": "prompt|image model|none",
 "why": "one or two sentences"}"""

ORIGIN_IDEAS = """You judge a picture made from a prompt that was INVENTED from an idea - a brief, a random roll of
ingredients, or a wildcard template with its picks. You get the IDEA, the PROMPT and the RENDER. Answer only from what you
see and read.

Return only JSON:
{"idea_in_prompt": 0-100,     // did the prompt keep the idea's ingredients and make one coherent picture of them?
 "idea_in_render": 0-100,     // does the render show the idea?
 "render_coherence": 0-100,
 "issues": ["..."],
 "likely_fault": "prompt|idea|image model|none",   // "idea" = the ingredients themselves do not go together
 "why": "one or two sentences"}"""


def _origin_kind(prov: Dict[str, Any]) -> str:
    variant = (prov.get("variant") or "").lower()
    if prov.get("mode") == "photobash" or variant.startswith("photobash") or len(prov.get("sources") or []) > 1:
        return "photobash"
    if variant.startswith("rewrite") and prov.get("source_prompt"):
        return "rewrite"
    if variant.startswith("ideas") and (prov.get("brief") or prov.get("roll") or prov.get("filled") or prov.get("template")):
        return "ideas"
    return ""


def _contributes(i: int, prov: Dict[str, Any]) -> str:
    takes = (prov.get("takes") or [])
    t = takes[i] if i < len(takes) else "everything"
    drawn = [a for a, n in (prov.get("assignment") or {}).items() if int(n) == i + 1]
    if drawn:
        return "drawn for: " + ", ".join(drawn)
    if isinstance(t, list) and t:
        return "only: " + ", ".join(t)
    return "everything"


def origin_check(checker, cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prov = cell.get("prov") or {}
    kind = _origin_kind(prov)
    if not kind:
        return None
    prompt = cell.get("prompt") or ""
    render = [{"type": "text", "text": "RENDER:"}, {"type": "image_url", "image_url": {"url": _image_url(cell["file"], 768)}}]
    if kind == "photobash":
        srcs = [s for s in (prov.get("sources") or []) if os.path.isfile(s)][:4]
        if len(srcs) < 2:
            return None
        weights = prov.get("weights") or []
        content = []
        for i, s in enumerate(srcs):
            w = f", weight {weights[i]}" if i < len(weights) else ""
            content += [{"type": "text", "text": f"SOURCE {i + 1}{w} - contributes {_contributes(i, prov)}:"},
                        {"type": "image_url", "image_url": {"url": _image_url(s, 640)}}]
        content += [{"type": "text", "text": "PROMPT:\n" + prompt}] + render
        out = checker._ask(ORIGIN_PHOTOBASH, content, 1500)
    elif kind == "rewrite":
        content = [{"type": "text", "text": f"OPERATION: {prov.get('op') or (prov.get('variant') or '').replace('rewrite', '').strip()}\n"
                                              f"INSTRUCTION: {prov.get('instruction') or '(none - format conversion)'}\n\n"
                                              f"ORIGINAL PROMPT:\n{prov.get('source_prompt')}"}]
        img = prov.get("source_image")
        if img and os.path.isfile(img):
            content += [{"type": "text", "text": "IMAGE THE ORIGINAL DESCRIBED:"}, {"type": "image_url", "image_url": {"url": _image_url(img, 640)}}]
        content += [{"type": "text", "text": "NEW PROMPT:\n" + prompt}] + render
        out = checker._ask(ORIGIN_REWRITE, content, 1000)
    else:
        idea = []
        if prov.get("brief"):
            idea.append("brief: " + str(prov["brief"]))
        if prov.get("roll"):
            r = prov["roll"]
            idea.append("rolled ingredients: " + (", ".join(f"{k}: {v}" for k, v in r.items()) if isinstance(r, dict) else str(r)))
        if prov.get("template"):
            idea.append("wildcard template: " + str(prov["template"]))
        if prov.get("picks"):
            idea.append("its random picks: " + "; ".join(prov["picks"]) if isinstance(prov["picks"], list) else str(prov["picks"]))
        elif prov.get("filled"):
            idea.append("filled template: " + str(prov["filled"]))
        content = [{"type": "text", "text": "IDEA:\n" + "\n".join(idea) + "\n\nPROMPT:\n" + prompt}] + render
        out = checker._ask(ORIGIN_IDEAS, content, 1000)
    out["kind"] = kind
    return out
