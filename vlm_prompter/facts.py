"""
Measured facts about the people in an image - detectors instead of a vision LLM's impression.

  pose      rtmlib Wholebody, mode "performance" = RTMW-x-l 384x288 (133 COCO-WholeBody keypoints) + YOLOX-m person
            detector. Facing direction and pose comparison: pose_geom.py.
  faces     UniFace 4.3: RetinaFace detection, AgeGender (age in years + gender), FairFace (age group, ethnicity),
            HeadPose (yaw / pitch / roll in degrees).
  emotion   HSEmotion ONNX (AffectNet-trained) on the face crop - UniFace's own emotion model needs PyTorch.

Everything runs on onnxruntime (CPU build installed; the GPU build is a drop-in swap). Models download on first use
into the libraries' own caches. Models load once per process (lazy, thread-safe).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

from . import pose_geom

MAX_SIDE = 1600
MIN_PERSON_HEIGHT = 0.20      # a person is described when their body is at least this fraction of the picture height
MAX_PERSONS = 6
_LOCK = threading.Lock()
_MODELS: Dict[str, Any] = {}


def _models() -> Dict[str, Any]:
    with _LOCK:
        if not _MODELS:
            from rtmlib import Wholebody
            from uniface import AgeGender, FairFace, HeadPose, RetinaFace
            _MODELS["pose"] = Wholebody(mode="performance", backend="onnxruntime", device="cpu")
            _MODELS["det"] = RetinaFace()
            _MODELS["agegender"] = AgeGender()
            _MODELS["fairface"] = FairFace()
            _MODELS["headpose"] = HeadPose()
            try:
                from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
                _MODELS["emotion"] = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
            except Exception as exc:  # optional: facts still work without expression
                print(f"WARNING: emotion model unavailable ({type(exc).__name__}: {exc})")
                _MODELS["emotion"] = None
        return _MODELS


def load_bgr(path: str) -> np.ndarray:
    """EXIF-corrected image as a BGR uint8 array (unicode-safe), long side capped at MAX_SIDE."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        rgb = np.asarray(im)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def age_words(age: Optional[float]) -> str:
    if age is None:
        return ""
    a = int(round(age))
    if a < 13:
        return "a child"
    if a < 20:
        return "late teens" if a >= 17 else "teenage"
    decade = (a // 10) * 10
    part = "early" if a % 10 <= 3 else "mid" if a % 10 <= 6 else "late"
    return f"{part} {decade}s"


def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def _face_facts(img: np.ndarray, face, m: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"bbox": [round(float(v), 1) for v in face.bbox], "confidence": round(float(face.confidence), 3)}
    try:
        ag = m["agegender"].predict(img, face)
        out["age"], out["gender"] = ag.age, ("female" if ag.gender == 0 else "male")
    except Exception as exc:
        out["age_error"] = str(exc)
    try:
        ff = m["fairface"].predict(img, face)
        out["age_group"], out["ethnicity"] = ff.age_group, ff.race
        out["gender_fairface"] = "female" if ff.gender == 0 else "male"
    except Exception as exc:
        out["fairface_error"] = str(exc)
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    crop_bgr = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    try:
        hp = m["headpose"].estimate(crop_bgr)                 # HeadPose takes a face crop, not image + box
        out["yaw"], out["pitch"], out["roll"] = round(float(hp.yaw), 1), round(float(hp.pitch), 1), round(float(hp.roll), 1)
    except Exception as exc:
        out["headpose_error"] = str(exc)
    if m.get("emotion") is not None:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad, ::-1]       # RGB crop
        if crop.size:
            try:
                label, scores = m["emotion"].predict_emotions(crop, logits=False)
                out["expression"] = str(label).lower()
                out["expression_conf"] = round(float(np.max(scores)), 3)
            except Exception as exc:
                out["emotion_error"] = str(exc)
    _reliability(out, img.shape[0])                 # last: needs gender, age, yaw AND expression
    return out


def _group_range(group: Optional[str]):
    if not group:
        return None
    g = str(group).replace("+", "-200")
    try:
        lo, hi = [int(x) for x in g.split("-")[:2]]
        return lo, hi
    except ValueError:
        return None


def _reliability(f: Dict[str, Any], img_h: int) -> None:
    """Keep attributes only where the models agree and the face is usable (Nov 13 test: profile faces gave
    'male 59' vs FairFace 'female 20-29', and age 25 vs group '3-9')."""
    h = f["bbox"][3] - f["bbox"][1]
    f["usable"] = h >= 0.06 * img_h and abs(f.get("yaw") or 0) <= 60
    g1, g2 = f.get("gender"), f.get("gender_fairface")
    f["gender_ok"] = bool(f["usable"] and g1 and g1 == g2)
    rng = _group_range(f.get("age_group"))
    age = f.get("age")
    f["age_ok"] = bool(f["usable"] and age is not None and rng and rng[0] - 7 <= age <= rng[1] + 7)
    f["ethnicity_ok"] = bool(f["usable"])
    f["expression_ok"] = bool(f["usable"] and (f.get("expression_conf") or 0) >= 0.4)


def analyze(path: str) -> Dict[str, Any]:
    """Detect people + faces; return facts for the main person and short facts for up to 3 others."""
    m = _models()
    img = load_bgr(path)
    kps, scores = m["pose"](img)
    kps = np.asarray(kps) if kps is not None else np.zeros((0, 133, 2))
    scores = np.asarray(scores) if scores is not None else np.zeros((0, 133))
    people = [i for i in range(len(kps)) if np.sum(scores[i][:17] >= pose_geom.MIN_SCORE) >= 6]
    res: Dict[str, Any] = {"size": [int(img.shape[1]), int(img.shape[0])], "people": len(people), "main": None,
                           "others": []}
    faces = m["det"].detect(img) or []
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[:4]
    face_facts = [_face_facts(img, f, m) for f in faces]
    res["faces"] = len(faces)
    H, W = img.shape[0], img.shape[1]
    persons = []
    for i in people:
        box = pose_geom.person_box(kps[i], scores[i])
        if not box:
            continue
        hf = (box[3] - box[1]) / H
        if hf < MIN_PERSON_HEIGHT:
            continue
        k, s = kps[i], scores[i]
        head = k[0] if s[0] >= pose_geom.MIN_SCORE else None
        face = None
        for ff in face_facts:
            b = ff["bbox"]
            if head is not None and b[0] <= head[0] <= b[2] and b[1] <= head[1] <= b[3]:
                face = ff
                break
        persons.append({"det_index": int(i), "box": [round(v, 1) for v in box], "height_frac": round(hf, 3),
                        "area_frac": round((box[2] - box[0]) * (box[3] - box[1]) / (W * H), 4),
                        "x": round((box[0] + box[2]) / 2 / W, 3), "facing": pose_geom.facing(k, s), "face": face,
                        "keypoints": k.round(1).tolist(), "scores": s.round(3).tolist()})
    persons = sorted(persons, key=lambda q: q["area_frac"], reverse=True)[:MAX_PERSONS]
    persons.sort(key=lambda q: q["x"])                                 # left to right
    for n, q in enumerate(persons, 1):
        q["n"] = n
        q["position"] = "left" if q["x"] < 0.36 else "right" if q["x"] > 0.64 else "center"
        q["dir"] = pose_geom.horiz_dir(q["facing"])
    res["persons"] = persons
    res["significant"] = len(persons)
    rels = []
    for a in range(len(persons)):
        for b in range(a + 1, len(persons)):
            cls = pose_geom.relation(persons[a]["dir"], persons[b]["dir"])
            if cls:
                rels.append({"a": persons[a]["n"], "b": persons[b]["n"], "class": cls})
    res["relations"] = rels
    main = pose_geom.main_person(kps[people], scores[people]) if people else None
    if main is not None:
        i = people[main]
        k, s = kps[i], scores[i]
        info: Dict[str, Any] = {"facing": pose_geom.facing(k, s), "keypoints": k.round(1).tolist(),
                                "scores": s.round(3).tolist()}
        # the face that sits on this person's head (nose / eyes inside the face box) or the biggest face
        head = k[0] if s[0] >= pose_geom.MIN_SCORE else None
        match = None
        for ff in face_facts:
            b = ff["bbox"]
            if head is not None and b[0] <= head[0] <= b[2] and b[1] <= head[1] <= b[3]:
                match = ff
                break
        if match is None and face_facts and len(people) == 1:
            match = face_facts[0]
        info["face"] = match
        res["main"] = info
        res["others"] = [ff for ff in face_facts if ff is not match][:3]
    elif face_facts:                       # face visible, body not detected (tight portrait crops)
        res["main"] = {"facing": None, "face": face_facts[0], "keypoints": None, "scores": None}
        res["others"] = face_facts[1:4]
    return res


def _person_words(q: Dict[str, Any], include_ethnicity: bool) -> str:
    face = q.get("face") or {}
    who = []
    if face.get("gender_ok"):
        who.append("a woman" if face["gender"] == "female" else "a man")
    if face.get("age_ok"):
        w = age_words(face["age"])
        who.append(w if w.startswith(("a ", "teen", "late teen")) else f"in the {w}")
    if include_ethnicity and face.get("ethnicity_ok") and face.get("ethnicity"):
        who.append(f"appears {face['ethnicity']}")
    if face.get("expression_ok"):
        who.append(f"expression {face['expression']}")
    if who and not who[0].startswith(("a woman", "a man", "a child")):
        who.insert(0, "a person")
    text = ", ".join(who) if who else "a person"
    f = q.get("facing") or {}
    if f.get("text") and f["text"] != "facing could not be determined":
        text += " - " + f["text"]
    return text


def facts_text(res: Dict[str, Any], include_ethnicity: bool = False) -> str:
    """Plain sentence(s) for a prompt hint or a report."""
    persons = (res or {}).get("persons") or []
    if len(persons) >= 2:
        n_all = max(res.get("people", 0), res.get("faces", 0))
        parts = [f"{len(persons)} main people, left to right:"]
        parts.append("; ".join(f"({q['n']}, {q['position']}) {_person_words(q, include_ethnicity)}" for q in persons) + ".")
        for r in res.get("relations") or []:
            txt = pose_geom.RELATION_TEXT.get(r["class"], "")
            if txt:
                txt = txt.replace("the left one", f"person {r['a']}").replace("the right one", f"person {r['b']}")
                parts.append(f"Persons {r['a']} and {r['b']}: {txt}." if not txt.startswith("person")
                             else txt[0].upper() + txt[1:] + ".")
        if n_all > len(persons):
            extra = n_all - len(persons)
            parts.append(f"{extra} more {'person' if extra == 1 else 'people'} smaller in the background.")
        return " ".join(parts)
    if not res or (not res.get("people") and not res.get("faces")):
        return "No people detected."
    n = max(res.get("people", 0), res.get("faces", 0))
    parts = [f"{n} person" + ("" if n == 1 else "s") + " in the picture."]
    main = res.get("main") or {}
    face = main.get("face") or {}
    who = []
    if face.get("gender_ok"):
        who.append("a woman" if face["gender"] == "female" else "a man")
    if face.get("age_ok"):
        words = age_words(face["age"])
        who.append(words if words.startswith(("a ", "teen", "late teen")) else f"in the {words}")
    if include_ethnicity and face.get("ethnicity_ok") and face.get("ethnicity"):
        who.append(f"appears {face['ethnicity']}")
    if face.get("expression_ok"):
        who.append(f"expression {face['expression']}")
    line = "Main person: " + (", ".join(who) if who else "person")
    if main.get("facing") and main["facing"].get("text"):
        line += "; " + main["facing"]["text"]
    if face.get("yaw") is not None:
        y = face["yaw"]
        if abs(y) >= 20:
            line += f"; head yaw {y:+.0f} degrees"
    parts.append(line + ".")
    return " ".join(parts)


# ===================================================================================================== cache
def cached_analyze(path: str) -> Dict[str, Any]:
    """analyze() with a persistent cache in review.sqlite (key: path + size + mtime) - detectors run once per file."""
    import json
    import os
    import sqlite3
    from .review import DB_PATH, _DB_LOCK
    st = os.stat(path)
    key = f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{int(st.st_mtime)}"
    with _DB_LOCK:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, path TEXT, data TEXT, created TEXT)")
            row = con.execute("SELECT data FROM facts WHERE key = ?", (key,)).fetchone()
        finally:
            con.close()
    if row:
        return json.loads(row[0])
    res = analyze(path)
    import datetime as _dt
    with _DB_LOCK:
        con = sqlite3.connect(DB_PATH, timeout=10)
        try:
            con.execute("INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?)",
                        (key, os.path.normcase(path), json.dumps(res), _dt.datetime.now().isoformat(timespec="seconds")))
            con.commit()
        finally:
            con.close()
    return res


def coarse_body(res: Dict[str, Any]) -> str:
    """toward | away | left | right | '' (unknown) - the main person's body direction, coarse."""
    f = ((res or {}).get("main") or {}).get("facing") or {}
    if f.get("body") == "away from camera":
        return "away"
    if f.get("turn") in ("profile", "three-quarter") and f.get("side"):
        return f["side"]
    if f.get("body") == "toward camera":
        return "toward"
    return ""


def coarse_head(res: Dict[str, Any]) -> str:
    """toward | away | left | right | '' - from head pose yaw when a usable face exists, else keypoints."""
    main = (res or {}).get("main") or {}
    face = main.get("face") or {}
    if face.get("usable") and face.get("yaw") is not None:
        y = face["yaw"]
        return "toward" if abs(y) < 20 else ("left" if y > 0 else "right")      # +yaw = turned to picture-left
    h = (main.get("facing") or {}).get("head", "")
    return {"toward camera": "toward", "away from camera": "away", "turned left": "left",
            "turned right": "right"}.get(h, "")


def pose_check(source: str, render: str) -> Dict[str, Any]:
    """Detector comparison of a render with its source: facing, mirror test, pose similarity, attribute drift."""
    s, r = cached_analyze(source), cached_analyze(render)
    out: Dict[str, Any] = {"source_facts": facts_text(s), "render_facts": facts_text(r),
                           "source_people": max(s.get("people", 0), s.get("faces", 0)),
                           "render_people": max(r.get("people", 0), r.get("faces", 0))}
    sb, rb, sh, rh = coarse_body(s), coarse_body(r), coarse_head(s), coarse_head(r)
    out["body"] = {"source": sb, "render": rb, "match": pose_geom.directions_agree(sb, rb) if sb and rb else None}
    out["head"] = {"source": sh, "render": rh, "match": pose_geom.directions_agree(sh, rh) if sh and rh else None}
    ms, mr = s.get("main") or {}, r.get("main") or {}
    if ms.get("keypoints") and mr.get("keypoints"):
        out["pose"] = pose_geom.compare_poses((np.array(ms["keypoints"]), np.array(ms["scores"])),
                                              (np.array(mr["keypoints"]), np.array(mr["scores"])))
    fs, fr = ms.get("face") or {}, mr.get("face") or {}
    drift: Dict[str, Any] = {}
    if fs.get("gender_ok") and fr.get("gender_ok") and fs["gender"] != fr["gender"]:
        drift["gender"] = f"{fs['gender']} -> {fr['gender']}"
    if fs.get("age_ok") and fr.get("age_ok") and abs(fs["age"] - fr["age"]) >= 10:
        drift["age"] = f"{fs['age']} -> {fr['age']}"
    if fs.get("expression_ok") and fr.get("expression_ok") and fs["expression"] != fr["expression"]:
        drift["expression"] = f"{fs['expression']} -> {fr['expression']}"
    if fs.get("ethnicity_ok") and fr.get("ethnicity_ok") and fs.get("ethnicity") != fr.get("ethnicity"):
        drift["ethnicity"] = f"{fs.get('ethnicity')} -> {fr.get('ethnicity')}"
    out["drift"] = drift
    problems = []
    if out["body"]["match"] is False:
        problems.append(f"body faces {sb} in the source, {rb} in the render")
    if out["head"]["match"] is False:
        problems.append(f"head turned {sh} in the source, {rh} in the render")
    if (out.get("pose") or {}).get("mirrored"):
        problems.append("the pose is a mirror image of the source")
    # several people: pair them left to right when the counts match, then compare direction and relations
    ps, pr = s.get("persons") or [], r.get("persons") or []
    out["persons"] = {"source": len(ps), "render": len(pr)}
    if len(ps) != len(pr) and (len(ps) >= 2 or len(pr) >= 2):
        problems.append(f"{len(ps)} main people in the source, {len(pr)} in the render")
    elif len(ps) >= 2:
        for a, b in zip(ps, pr):
            if a["dir"] and b["dir"] and not pose_geom.directions_agree(a["dir"], b["dir"]):
                problems.append(f"person {a['n']} ({a['position']}) is turned {a['dir']} in the source, {b['dir']} in the render")
        rs = {(x["a"], x["b"]): x["class"] for x in s.get("relations") or []}
        rr = {(x["a"], x["b"]): x["class"] for x in r.get("relations") or []}
        for key, cls in rs.items():
            if not pose_geom.relations_agree(cls, rr.get(key, "")):
                was = pose_geom.RELATION_TEXT.get(cls, cls)
                now = pose_geom.RELATION_TEXT.get(rr.get(key, ""), "no clear relation")
                problems.append(f"persons {key[0]} and {key[1]}: source - {was}; render - {now}")
    out["problems"] = problems
    return out


# ===================================================================================================== for prompt writing
# Nov 13 audit verdicts on the test folder: the detectors beat the vision model only on HEAD direction (12:5);
# the model was better on gender, age, ethnicity, people count and body toward/away. So prompt writing gets head
# direction (and the relations it implies) only.
_HEAD_WORDS = {"toward": "faces the viewer", "away": "has the head turned away from the viewer",
               "left": "has the head turned toward the left side of the picture",
               "right": "has the head turned toward the right side of the picture"}


def person_head(q: Dict[str, Any]) -> str:
    """toward | away | left | right | '' for one person: head-pose yaw when the face is usable, else keypoints."""
    face = q.get("face") or {}
    if face.get("usable") and face.get("yaw") is not None:
        y = face["yaw"]
        return "toward" if abs(y) < 20 else ("left" if y > 0 else "right")      # +yaw = turned to picture-left
    h = (q.get("facing") or {}).get("head", "")
    return {"toward camera": "toward", "away from camera": "away", "turned left": "left",
            "turned right": "right"}.get(h, "")


def direction_text(res: Dict[str, Any]) -> str:
    """Head direction of each main person + relations between them, or '' when nothing is known."""
    persons = (res or {}).get("persons") or []
    known = [(q, person_head(q)) for q in persons]
    known = [(q, d) for q, d in known if d]
    if not known:
        return ""
    if len(persons) == 1:
        return f"The person {_HEAD_WORDS[known[0][1]]}."
    parts = ["Main people from left to right: " + "; ".join(
        f"person {q['n']} ({q['position']}) {_HEAD_WORDS[d]}" for q, d in known) + "."]
    dirs = {q["n"]: ("viewer" if d == "toward" else d) for q, d in known}
    for a in range(1, len(persons)):                   # neighbours only - distant pairs are not an interaction
        b = a + 1
        if a in dirs and b in dirs:
            cls = pose_geom.relation(dirs[a], dirs[b])
            if dirs[a] == dirs[b]:                     # both looking the same way: their own directions say it all
                cls = ""
            if cls in ("each_other", "away_from_each_other", "a_toward_b", "b_toward_a"):
                if True:
                    txt = pose_geom.RELATION_TEXT[cls].replace("the left one", f"person {a}").replace("the right one", f"person {b}")
                    parts.append((f"Persons {a} and {b} {txt}." if not txt.startswith("person") else txt[0].upper() + txt[1:] + "."))
    return " ".join(parts)
