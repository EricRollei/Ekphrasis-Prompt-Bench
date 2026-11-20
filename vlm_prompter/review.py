"""
Prompt Review (phase 1): load renders from Krea 2 sweep folders or any image folder, find each render's prompt,
link the prompt back to how it was made (VLM model, preset, instructions, source image), and store ratings.

Spec: docs/SPEC_review.md (v0.2). Phase 2 (vision checks) and 3 (reports) build on the tables here.

Sources
  sweep   a folder under settings["sweeps_dir"] with sweep_manifest_*.json (Eric_Krea2 Sweep Plan output):
          cells[i] = {file, overrides{prompt, checkpoint, ...}, seed, loras, time_s, metrics, label, ...}
  folder  any image folder (settings["review_folders"]); the prompt is read from the image itself
          (PNG chunks sweep_cell / krea2_settings / parameters / ComfyUI prompt; JPEG/WebP EXIF + XMP) or a sidecar
          <stem>.prompt.txt / <stem>.txt.

Ratings live in review.sqlite (app folder), keyed by the render's full path:
  stars 1-5, keeper 0/1, defects (list of DEFECT ids), adherence 0-100, fidelity 0-100 (manual for now;
  phase 2 adds the vision values beside them), note.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import csv
import datetime as _dt
import glob
import json
import os
import re
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from . import config
from .images import is_image, natural_key
from .outputs import prompt_sha, unique_path
from .store import load_instructions

DB_PATH = os.path.join(config.APP_DIR, "review.sqlite")
_DB_LOCK = threading.Lock()

# (id, key, label) - clear-cut defects, yes/no. Keys are the keyboard shortcuts in the Review tab.
DEFECTS = [
    ("signature", "s", "signature / watermark"),
    ("text", "t", "unwanted text"),
    ("nudity", "n", "unwanted nudity"),
    ("anatomy", "a", "anatomy error"),
    ("physics", "p", "physics error"),
    ("subject", "e", "extra / missing main subject"),
]
DEFECT_IDS = {d[0] for d in DEFECTS}
FOLDER_LIMIT = 3000          # images read from one plain folder (metadata reading is per file)


# ===================================================================================================== sources
def list_sweeps(sweeps_dir: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Sweep folders, newest first. Cheap: no manifest parsing (some manifests are large)."""
    if not sweeps_dir or not os.path.isdir(sweeps_dir):   # unset sweeps folder
        return []
    if not sweeps_dir or not os.path.isdir(sweeps_dir):
        return []
    out = []
    for e in os.scandir(sweeps_dir):
        if not e.is_dir():
            continue
        try:
            names = os.listdir(e.path)
        except OSError:
            continue
        if not any(n.startswith("sweep_manifest") and n.endswith(".json") for n in names):
            continue
        out.append({"kind": "sweep", "path": e.path, "name": e.name, "mtime": e.stat().st_mtime,
                    "images": sum(1 for n in names if n.startswith("cell_") and is_image(n))})
    out.sort(key=lambda s: s["mtime"], reverse=True)
    out = out[:limit]
    counts = rated_counts([s["path"] for s in out])
    for s in out:
        s["rated"] = counts.get(os.path.normcase(s["path"]), 0)
    return out


# ===================================================================================================== prompt from images
_PROMPT_KEYS = ("prompt", "positive", "positive_prompt", "sweep_prompt", "text", "caption")


def _longest_prompt_in(obj: Any, depth: int = 0) -> str:
    """Longest string under a prompt-like key anywhere in a JSON structure."""
    best = ""
    if depth > 8:
        return best
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.lower() in _PROMPT_KEYS and len(v) > len(best) and len(v) >= 12:
                best = v
            elif isinstance(v, (dict, list)):
                cand = _longest_prompt_in(v, depth + 1)
                if len(cand) > len(best):
                    best = cand
    elif isinstance(obj, list):
        for v in obj:
            cand = _longest_prompt_in(v, depth + 1)
            if len(cand) > len(best):
                best = cand
    return best


def _find_key(obj: Any, names: Tuple[str, ...], depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in names and isinstance(v, (str, int, float)) and v != "":
                return v
        for v in obj.values():
            r = _find_key(v, names, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, names, depth + 1)
            if r is not None:
                return r
    return None


def _json(text: Any) -> Any:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _user_comment(raw: Any) -> str:
    if isinstance(raw, bytes):
        for head, enc in ((b"ASCII\x00\x00\x00", "ascii"), (b"UNICODE\x00", "utf-16"), (b"JIS\x00\x00\x00\x00\x00", "shift_jis")):
            if raw.startswith(head):
                return raw[len(head):].decode(enc, "replace").strip("\x00 ").strip()
        return raw.decode("utf-8", "replace").strip("\x00 ").strip()
    return str(raw or "").strip()


def read_image_prompt(path: str) -> Tuple[str, Dict[str, Any]]:
    """(prompt, info) from an image's metadata or a sidecar file. info may hold checkpoint / seed / source."""
    info: Dict[str, Any] = {}
    prompt = ""
    try:
        with Image.open(path) as im:
            meta = dict(im.info)
            cell = _json(meta.get("sweep_cell"))
            if isinstance(cell, dict) and isinstance(cell.get("prompt"), str):
                prompt, info["source"] = cell["prompt"], "sweep_cell chunk"
            k2 = _json(meta.get("krea2_settings"))
            if k2 is not None:
                if not prompt:
                    prompt = _longest_prompt_in(k2)
                    info["source"] = "krea2_settings chunk" if prompt else ""
                ck = _find_key(k2, ("checkpoint", "transformer_path", "ckpt_name", "model_path"))
                if ck:
                    info["checkpoint"] = str(ck)
                sd = _find_key(k2, ("seed",))
                if sd is not None:
                    info["seed"] = sd
            if not prompt and isinstance(meta.get("parameters"), str):            # A1111 / Forge
                prompt = meta["parameters"].split("\nNegative prompt:")[0].strip()
                info["source"] = "parameters"
            if not prompt and meta.get("prompt"):                                 # ComfyUI API graph
                graph = _json(meta.get("prompt"))
                if isinstance(graph, dict):
                    texts = []
                    for node in graph.values():
                        if not isinstance(node, dict):
                            continue
                        ct = str(node.get("class_type", ""))
                        if any(s in ct for s in ("TextEncode", "Prompt", "Text")):
                            for kk, vv in (node.get("inputs") or {}).items():
                                if isinstance(vv, str) and kk.lower() in ("text", "prompt", "positive", "text_g", "text_l"):
                                    texts.append(vv)
                    if texts:
                        prompt, info["source"] = max(texts, key=len), "ComfyUI prompt graph"
            if not prompt:                                                        # JPEG / WebP: EXIF, XMP
                try:
                    exif = im.getexif()
                    desc = exif.get(0x010E)
                    uc = exif.get_ifd(0x8769).get(0x9286)
                    for cand in (_user_comment(uc), str(desc or "").strip()):
                        if len(cand) >= 12:
                            prompt, info["source"] = cand, "EXIF"
                            break
                except Exception:
                    pass
            if not prompt:
                xmp = meta.get("xmp") or meta.get("XML:com.adobe.xmp")
                if isinstance(xmp, bytes):
                    xmp = xmp.decode("utf-8", "replace")
                if isinstance(xmp, str):
                    m = re.search(r"<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>", xmp, re.S)
                    if m and len(m.group(1).strip()) >= 12:
                        prompt, info["source"] = m.group(1).strip(), "XMP"
    except Exception as exc:  # unreadable image: report, keep going
        info["error"] = f"{type(exc).__name__}: {exc}"
    if not prompt:
        stem = os.path.splitext(path)[0]
        for side in (stem + ".prompt.txt", stem + ".txt"):
            if os.path.isfile(side):
                try:
                    with open(side, "r", encoding="utf-8", errors="replace") as handle:
                        text = handle.read().strip()
                    if text:
                        prompt, info["source"] = text.splitlines()[0].strip(), os.path.basename(side)
                        break
                except OSError:
                    pass
    return prompt.strip(), info


# ===================================================================================================== provenance
class IndexLookup:
    """saved_index.jsonl (every prompt this app saved) by normalised prompt text; reloads when the file changes."""

    def __init__(self, runs_dir: str):
        self.path = os.path.join(runs_dir, "saved_index.jsonl")
        self.runs_dir = runs_dir
        self._mtime = None
        self._by_text: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _refresh(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._by_text, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        by: Dict[str, Dict[str, Any]] = {}
        with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                key = self.norm(str(r.get("prompt", "")))
                if key and (r.get("model") or key not in by):      # prefer records that carry a model
                    by[key] = r
        self._by_text, self._mtime = by, mtime

    def find(self, prompt: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._refresh()
            r = self._by_text.get(self.norm(prompt))
        if not r:
            return None
        ins = load_instructions(self.runs_dir, r.get("instr", "")) if r.get("instr") else None
        mods = [(m[0] + ("=" + m[1] if len(m) > 1 and m[1] else "")) for m in (ins or {}).get("modifiers", []) or []]
        return {"model": r.get("model", ""), "preset": r.get("preset", ""), "instr": r.get("instr", ""),
                "source_image": r.get("image", ""), "made": r.get("made", "") or r.get("saved", ""),
                "modifiers": mods, "note": (ins or {}).get("note", ""), "variant": r.get("variant", ""),
                "mode": r.get("mode", ""), "sources": list(r.get("sources") or [])}      # Nov 20: photobash = several


_INDEX: Optional[IndexLookup] = None


def index_lookup() -> IndexLookup:
    global _INDEX
    if _INDEX is None:
        _INDEX = IndexLookup(config.RUNS_DIR)
    return _INDEX


# ===================================================================================================== loading sources
def _short_ckpt(value: Any) -> str:
    if not value:
        return ""
    name = os.path.basename(str(value).replace("\\\\", "\\"))
    return os.path.splitext(name)[0]


def _cell_record(full: str, prompt: str, column: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    sha = prompt_sha(prompt) if prompt else ""
    rec = {"file": full, "name": os.path.basename(full), "prompt": prompt, "sha": sha, "col": column}
    rec.update(extra)
    rec["prov"] = index_lookup().find(prompt) if prompt else None
    return rec


def load_sweep(path: str) -> Dict[str, Any]:
    mans = sorted(glob.glob(os.path.join(path, "sweep_manifest*.json")))
    if not mans:
        raise FileNotFoundError(f"no sweep_manifest*.json in {path}")
    with open(mans[0], "r", encoding="utf-8") as handle:
        m = json.load(handle)
    base = m.get("base") or {}
    cells = []
    for c in m.get("cells", []):
        f = c.get("file")
        if not f:
            continue
        full = os.path.join(path, f)
        if not os.path.isfile(full):
            continue
        ov = c.get("overrides") or {}
        prompt = ov.get("prompt") or base.get("prompt") or ""
        if not prompt:
            prompt, _ = read_image_prompt(full)
        ckpt = ov.get("checkpoint") or base.get("checkpoint") or ""
        col_parts = [f"{k}={_short_ckpt(v) if k == 'checkpoint' else v}" for k, v in ov.items() if k != "prompt"]
        if c.get("loras"):
            col_parts.append(f"loras={c['loras']}")
        metrics = c.get("metrics") or {}
        cells.append(_cell_record(full, prompt, "  ".join(col_parts) or f"cell {c.get('index', '')}", {
            "index": c.get("index"), "checkpoint": _short_ckpt(ckpt), "seed": c.get("seed"),
            "loras": c.get("loras", ""), "time_s": c.get("time_s"),
            "reference": bool(ov.get("lora_baseline")),          # Eric_Krea2: the no-(test-)LoRA reference cell
            "clip": metrics.get("clip_fraction"), "sharp": metrics.get("sharpness_tenengrad"),
        }))
    _mark_references(cells)
    about = m.get("about") or {}
    return {"kind": "sweep", "path": path, "name": os.path.basename(path), "sweep_name": m.get("sweep_name", ""),
            "created": m.get("created", ""), "notes": about.get("run_notes", "") or about.get("preset_notes", ""),
            "cells": _attach_all(cells)}


def _mark_references(cells: List[Dict[str, Any]]) -> None:
    """Older sweeps without lora_baseline: in each prompt + seed + checkpoint group, the cell with strictly the fewest
    LoRAs is the reference (always-on panel LoRAs are in every cell, so 'fewest', not 'none')."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault((c.get("sha"), str(c.get("seed")), c.get("checkpoint")), []).append(c)
    for g in groups.values():
        if len(g) < 2 or any(c.get("reference") for c in g):
            continue
        n = [len([x for x in str(c.get("loras") or "").splitlines() if x.strip()]) for c in g]
        low = min(n)
        if n.count(low) == 1 and max(n) > low:
            g[n.index(low)]["reference"] = True


def load_folder(path: str, recursive: bool = True) -> Dict[str, Any]:
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted((d for d in dirnames if not d.startswith((".", "_"))), key=natural_key) if recursive else []
        for n in sorted(filenames, key=natural_key):
            if is_image(n):
                files.append(os.path.join(dirpath, n))
                if len(files) >= FOLDER_LIMIT:
                    break
        if len(files) >= FOLDER_LIMIT:
            break
    cells = []
    for full in files:
        prompt, info = read_image_prompt(full)
        rel_dir = os.path.relpath(os.path.dirname(full), path)
        col = _short_ckpt(info.get("checkpoint")) or ("" if rel_dir == "." else rel_dir)
        cells.append(_cell_record(full, prompt, col or os.path.basename(full), {
            "checkpoint": _short_ckpt(info.get("checkpoint")), "seed": info.get("seed"),
            "prompt_source": info.get("source", ""), "read_error": info.get("error", ""),
        }))
    return {"kind": "folder", "path": path, "name": os.path.basename(path.rstrip("\\/")) or path,
            "truncated": len(files) >= FOLDER_LIMIT, "limit": FOLDER_LIMIT, "cells": _attach_all(cells)}


def _attach_all(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ratings + vision check summaries (vision imports review, so it is imported here, not at the top)."""
    from .vision import attach_checks
    from .prompt_grades import grades_for
    cells = attach_checks(attach_ratings(cells))
    grades = grades_for(c.get("sha") for c in cells)
    for c in cells:
        c["pgrade"] = grades.get(c.get("sha"))
    return cells


# ===================================================================================================== ratings DB
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS ratings (
        file TEXT PRIMARY KEY, source TEXT, prompt_sha TEXT, stars INTEGER, keeper INTEGER DEFAULT 0,
        defects TEXT DEFAULT '[]', adherence INTEGER, fidelity INTEGER, note TEXT DEFAULT '', updated TEXT)""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(ratings)")}
    if "lesson" not in cols:                       # added Nov 13: note marked as a prompt-writing lesson
        con.execute("ALTER TABLE ratings ADD COLUMN lesson INTEGER DEFAULT 0")
    con.execute("CREATE INDEX IF NOT EXISTS ratings_source ON ratings(source)")
    con.execute("CREATE INDEX IF NOT EXISTS ratings_sha ON ratings(prompt_sha)")
    return con


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def attach_ratings(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cells:
        return cells
    keys = [_key(c["file"]) for c in cells]
    found: Dict[str, Dict[str, Any]] = {}
    with _DB_LOCK:
        con = _db()
        try:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT file, stars, keeper, defects, adherence, fidelity, note, lesson FROM ratings WHERE file IN ({','.join('?' * len(chunk))})"
                for row in con.execute(q, chunk):
                    found[row[0]] = {"stars": row[1], "keeper": bool(row[2]), "defects": json.loads(row[3] or "[]"),
                                     "adherence": row[4], "fidelity": row[5], "note": row[6] or "",
                                     "lesson": bool(row[7])}
        finally:
            con.close()
    for c, k in zip(cells, keys):
        c["rating"] = found.get(k)
    return cells


def rated_counts(sources: Iterable[str]) -> Dict[str, int]:
    keys = [os.path.normcase(s) for s in sources]
    if not keys:
        return {}
    out: Dict[str, int] = {}
    with _DB_LOCK:
        con = _db()
        try:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = (f"SELECT source, COUNT(*) FROM ratings WHERE source IN ({','.join('?' * len(chunk))}) "
                     "AND (stars IS NOT NULL OR keeper = 1 OR defects != '[]') GROUP BY source")
                out.update({r[0]: r[1] for r in con.execute(q, chunk)})
        finally:
            con.close()
    return out


def set_rating(file: str, source: str, sha: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """Upsert only the fields given. stars None/0 clears; defects is a full list."""
    clean: Dict[str, Any] = {}
    if "stars" in fields:
        s = fields["stars"]
        clean["stars"] = int(s) if s not in (None, "", 0, "0") else None
        if clean["stars"] is not None and not 1 <= clean["stars"] <= 5:
            raise ValueError("stars must be 1-5")
    if "keeper" in fields:
        clean["keeper"] = 1 if fields["keeper"] else 0
    if "lesson" in fields:
        clean["lesson"] = 1 if fields["lesson"] else 0
    if "defects" in fields:
        bad = [d for d in fields["defects"] if d not in DEFECT_IDS]
        if bad:
            raise ValueError(f"unknown defect(s): {bad}")
        clean["defects"] = json.dumps(sorted(set(fields["defects"])))
    for k in ("adherence", "fidelity"):
        if k in fields:
            v = fields[k]
            clean[k] = None if v in (None, "") else max(0, min(100, int(v)))
    if "note" in fields:
        clean["note"] = str(fields["note"] or "")
    clean["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
    key = _key(file)
    with _DB_LOCK:
        con = _db()
        try:
            con.execute("INSERT OR IGNORE INTO ratings (file, source, prompt_sha) VALUES (?, ?, ?)",
                        (key, os.path.normcase(source or ""), sha or ""))
            sets = ", ".join(f"{k} = ?" for k in clean) + ", source = ?, prompt_sha = COALESCE(NULLIF(?, ''), prompt_sha)"
            con.execute(f"UPDATE ratings SET {sets} WHERE file = ?",
                        list(clean.values()) + [os.path.normcase(source or ""), sha or "", key])
            con.commit()
        finally:
            con.close()
    return attach_ratings([{"file": file}])[0]["rating"]


# ===================================================================================================== export
def export_csv(source: Dict[str, Any]) -> str:
    """Ratings + provenance of one loaded source -> review_ratings_<date>.csv in that folder (never overwrites)."""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    path = unique_path(os.path.join(source["path"], f"review_ratings_{stamp}.csv"))
    cols = ["file", "column", "checkpoint", "seed", "stars", "keeper", "defects", "adherence", "fidelity", "note", "lesson",
            "vision_adherence", "vision_fidelity", "vision_defects",
            "vlm_model", "preset", "instructions", "modifiers", "source_image", "prompt"]
    with open(path, "x", encoding="utf-8-sig", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(cols)
        for c in source["cells"]:
            r = c.get("rating") or {}
            p = c.get("prov") or {}
            w.writerow([c["name"], c.get("col", ""), c.get("checkpoint", ""), c.get("seed", ""),
                        r.get("stars") or "", "yes" if r.get("keeper") else "", ";".join(r.get("defects") or []),
                        r.get("adherence") if r.get("adherence") is not None else "",
                        r.get("fidelity") if r.get("fidelity") is not None else "", r.get("note", ""),
                        "yes" if r.get("lesson") else "",
                        (c.get("check") or {}).get("adherence", ""), (c.get("check") or {}).get("fidelity", ""),
                        ";".join((c.get("check") or {}).get("defects") or []),
                        p.get("model", ""), p.get("preset", ""), p.get("instr", ""), ";".join(p.get("modifiers") or []),
                        p.get("source_image", ""), c.get("prompt", "")])
    return path
