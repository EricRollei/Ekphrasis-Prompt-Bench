"""
The GUI: a small local web server (127.0.0.1 only) + app.html, same pattern as the LoRA Catalog editor.

    python -m vlm_prompter serve            then open http://127.0.0.1:8790 (opens automatically)

Endpoints (JSON unless noted)
    GET  /                          app.html (read fresh on every request - edit and reload)
    GET  /api/state                 settings subset, presets, modifiers, queue
    GET  /api/models?all=0|1        models with favourite / loaded flags (favourites first)
    POST /api/favorite              {model, on}
    GET  /api/drives                drive letters
    GET  /api/ls?path=              folders + images of one folder: images = [{n: name, p: 1 if a .prompt.txt exists}]
    GET  /thumb?path=&side=256      JPEG thumbnail (disk-cached under .cache/thumbs)
    POST /api/expand                {paths, recursive} -> [{image, root}]
    POST /api/upload                raw file body, headers X-Filename (relative path ok) + X-Batch -> {image, root}
    GET  /api/queue  POST /api/queue {items}           the working queue (autosaved)
    GET  /api/queues                 named queues      POST /api/queues/save {name, items}   GET /api/queues/load?name=
    GET  /api/preset?id=            raw yaml           POST /api/preset/save {id, text} (old version kept in presets/.bak)
    POST /api/compose               {preset, mods, note, variations} -> composed system + user
    POST /api/run                   {items, model, preset, mods, note, notes, variations, reasoning, max_tokens,
                                     temperature, seed, force} -> {job}
    GET  /api/job?since=N           {state, events[N:]}        POST /api/job/cancel
    POST /api/plan                  {items, group_by, overrides} -> group file names
    POST /api/export                {items:[{image, root, prompts}], modes, format, group_by, group_dir,
                                     combined_path, overrides, remember} -> written files
    POST /api/open                  {path} open a folder in Explorer (selects the file if a file)

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import string
import subprocess
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import yaml
from PIL import Image, ImageOps

from . import config
from .images import IMAGE_EXTS, discover, is_image, is_media, is_video, load_rgb, natural_key, video_poster
from .instructions import compose, load_modifiers, load_presets, parse_modifier_args
from .jobs import USE_LOADED, JobSpec, make_client, model_modes, plan_outputs, run_generation, write_outputs

HERE = os.path.dirname(os.path.abspath(__file__))
THUMB_DIR = os.path.join(config.CACHE_DIR, "thumbs")
QUEUE_PATH = os.path.join(config.CACHE_DIR, "queue.json")
QUEUES_DIR = os.path.join(config.APP_DIR, "queues")
STAGING_DIR = os.path.join(config.APP_DIR, "staging")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9 _.\-()]+")

SETTINGS_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------------- job holder

def _list_opts(body):
    """Shared list settings sent by the GUI (Generate's Save tab values)."""
    return {"fmt": body.get("fmt", "lines"), "on_exists": body.get("on_exists", "append"),
            "dedupe": bool(body.get("dedupe", True)), "sidecar": bool(body.get("sidecar", True))}

class JobHolder:
    """One generation job at a time; events kept so the page can poll with ?since=N."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: List[Dict] = []
        self.state = "idle"                 # idle | running | done | cancelled | error
        self.cancel = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self, settings: Dict, spec: JobSpec) -> None:
        with self.lock:
            if self.state == "running":
                raise RuntimeError("a job is already running")
            self.events, self.state = [], "running"
            self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(settings, spec), daemon=True)
        self.thread.start()

    def _emit(self, event: Dict) -> None:
        event = {k: v for k, v in event.items() if k != "log_obj"}
        with self.lock:
            self.events.append(event)

    def _run(self, settings: Dict, spec: JobSpec) -> None:
        try:
            summary = run_generation(settings, spec, self._emit, self.cancel)
            summary["log_obj"].write({"event": "end", "errors": summary["errors"], "elapsed_s": summary["elapsed_s"]})
            self.state = "cancelled" if self.cancel.is_set() else "done"
        except Exception as exc:  # report everything to the page
            self._emit({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            self.state = "error"

    def poll(self, since: int) -> Dict:
        with self.lock:
            return {"state": self.state, "events": self.events[since:], "next": len(self.events)}


JOB = JobHolder()


# ----------------------------------------------------------------------------------------------------- helpers
def _thumb(path: str, side: int) -> bytes:
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{side}".encode("utf-8")).hexdigest()
    cached = os.path.join(THUMB_DIR, key[:2], key + ".jpg")
    if os.path.isfile(cached):
        with open(cached, "rb") as handle:
            return handle.read()
    img = video_poster(path) if is_video(path) else Image.open(path)
    if getattr(img, "format", None) == "JPEG":                      # decode at 1/2..1/8 scale straight from the JPEG (much faster)
        img.draft("RGB", (side * 2, side * 2))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (128, 128, 128))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((side, side), Image.BILINEAR, reducing_gap=2.0)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    data = buf.getvalue()
    os.makedirs(os.path.dirname(cached), exist_ok=True)
    with open(cached, "wb") as handle:
        handle.write(data)
    return data


def _drives() -> List[str]:
    if os.name != "nt":
        return ["/"]
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def _ls(path: str, suffix: str, json_suffix: str = ".prompt.json") -> Dict:
    path = os.path.abspath(path.strip().strip('"').strip("'"))
    select = ""
    if os.path.isfile(path):                     # a pasted file path opens its folder with the file selected
        path, select = os.path.dirname(path), os.path.basename(path)
    dirs, images = [], []
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        return {"path": path, "error": str(exc), "dirs": [], "images": [], "parent": os.path.dirname(path)}
    names = {e.name.lower() for e in entries}
    for e in sorted(entries, key=lambda e: natural_key(e.name)):
        try:
            if e.is_dir():
                if not e.name.startswith(("$", ".")) and e.name != "System Volume Information":
                    dirs.append({"name": e.name, "path": e.path})
            elif is_media(e.name):                  # slim rows: n = name, p = has a prompt file (1/0), v = video
                stem = os.path.splitext(e.name)[0]
                has = (stem + suffix).lower() in names or (stem + json_suffix).lower() in names
                row = {"n": e.name, "p": 1 if has else 0}
                if is_video(e.name):
                    row["v"] = 1
                images.append(row)
        except OSError:
            continue
    parent = os.path.dirname(path)
    return {"path": path, "sep": os.sep, "parent": parent if parent != path else "", "dirs": dirs, "images": images,
            "select": select}


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _remember_folder(settings: Dict, folder: str) -> None:
    recent = [f for f in settings.get("recent_folders", []) if os.path.normcase(f) != os.path.normcase(folder)]
    settings["recent_folders"] = ([folder] + recent)[:12]


def _items(body: Dict) -> List[tuple]:
    return [(it["image"], it.get("root") or os.path.dirname(it["image"])) for it in body.get("items", [])]


# ----------------------------------------------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    server_version = "VLMPrompter/0.2"

    def log_message(self, fmt, *args):            # quiet console
        pass

    # --- responses
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html")) else
                         "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"), "application/json")

    def _error(self, msg: str, code: int = 400) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> Dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    # --- routing
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            route = {
                "/": self.page, "/audit": self.audit_page, "/report": self.report_page, "/library": self.library_page,
                "/media": self.media,
                "/api/pools": self.pools_list, "/api/pools/load": self.pools_load,
                "/api/render/providers": self.r_providers, "/api/render/caps": self.r_caps,
                "/asset": self.asset,
                "/api/library": self.library_api, "/api/library/forensics": self.library_forensics, "/api/lessons": self.lessons_list,
                "/api/grade_suggestions": self.grade_suggestions, "/api/state": self.state,
                "/api/review/report": self.review_report,
                "/api/audit/results": self.audit_results, "/api/audit/status": self.audit_status,
                "/api/compare/status": self.compare_status, "/api/photobash/status": self.pb_status,
                "/api/ideas/status": self.id_status, "/api/ideas/wildcards": self.id_wildcards,
                "/api/rewrite/status": self.rw_status, "/api/settings/all": self.settings_all,
                "/api/lists/peek": self.lists_peek, "/api/wc/search": self.wc_search, "/api/wc/entries": self.wc_entries,
                "/api/render/state": self.r_state, "/api/render/status": self.r_status,
                "/api/render/fields": self.r_fields, "/api/render/for": self.r_for, "/api/models": self.models, "/api/drives": self.drives,
                "/api/ls": self.ls, "/thumb": self.thumb, "/api/queue": self.queue_get, "/api/queues": self.queues,
                "/api/queues/load": self.queues_load, "/api/preset": self.preset_get, "/api/job": self.job,
                "/api/lists": self.lists, "/api/modes": self.modes,
                "/api/review/sources": self.review_sources, "/api/review/load": self.review_load,
                "/api/review/check/status": self.check_status, "/api/review/check/detail": self.check_detail,
            }.get(url.path)
            if route is None:
                return self._error("not found", 404)
            route(q)
        except Exception as exc:
            traceback.print_exc()
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/upload":
                return self.upload()
            body = self._body()
            route = {
                "/api/favorite": self.favorite, "/api/expand": self.expand, "/api/queue": self.queue_put,
                "/api/queues/save": self.queues_save, "/api/preset/save": self.preset_save,
                "/api/compose": self.compose_preview, "/api/run": self.run, "/api/job/cancel": self.job_cancel,
                "/api/plan": self.plan, "/api/export": self.export, "/api/open": self.open_path,
                "/api/settings": self.settings_put, "/api/whence": self.whence,
                "/api/load": self.load_model, "/api/eject": self.eject_model,
                "/api/review/folders": self.review_folders, "/api/review/rate": self.review_rate,
                "/api/review/export": self.review_export,
                "/api/review/check/start": self.check_start, "/api/review/check/stop": self.check_stop,
                "/api/audit/start": self.audit_start, "/api/audit/stop": self.audit_stop,
                "/api/audit/verdict": self.audit_verdict,
                "/api/compare/start": self.compare_start, "/api/compare/stop": self.compare_stop,
                "/api/photobash/start": self.pb_start, "/api/photobash/stop": self.pb_stop,
                "/api/photobash/save": self.pb_save, "/api/photobash/preview": self.pb_preview,
                "/api/ideas/start": self.id_start, "/api/ideas/stop": self.id_stop, "/api/ideas/save": self.id_save,
                "/api/rewrite/start": self.rw_start, "/api/rewrite/stop": self.rw_stop, "/api/rewrite/save": self.rw_save,
                "/api/rewrite/load": self.rw_load, "/api/rewrite/split": self.rw_split,
                "/api/render/start": self.r_start, "/api/render/stop": self.r_stop, "/api/wc/roll": self.wc_roll,
                "/api/render/apikey": self.r_apikey, "/api/pgrade": self.pgrade, "/api/layout": self.layout_from_image,
                "/api/pools/save": self.pools_save, "/api/pools/delete": self.pools_delete, "/api/lessons/propose": self.lessons_propose,
                "/api/lessons/status": self.lessons_status, "/api/pgrades": self.pgrades, "/api/library/export": self.library_export,
                "/api/render/fromlast": self.r_fromlast, "/api/render/map": self.r_map,
                "/api/render/preset": self.r_preset, "/api/render/for_many": self.r_for_many,
            }.get(url.path)
            if route is None:
                return self._error("not found", 404)
            route(body)
        except Exception as exc:
            traceback.print_exc()
            self._error(f"{type(exc).__name__}: {exc}", 500)

    # --- pages and state
    def page(self, q):
        with open(os.path.join(HERE, "app.html"), "rb") as handle:
            self._send(200, handle.read(), "text/html; charset=utf-8")

    def state(self, q):
        s = config.load_settings()
        presets = load_presets(config.PRESETS_DIR)
        mods = load_modifiers(config.MODIFIERS_DIR)
        self._json({
            "settings": {k: s[k] for k in ("last_model", "favorite_models", "default_preset", "group_dir",
                                            "recent_folders", "per_image_suffix", "per_image_suffix_json",
                                            "load_context_length", "lmstudio_host", "max_side")},
            "drives": _drives(),
            "presets": [{"id": p.id, "name": p.name, "reasoning": p.reasoning, "max_tokens": p.max_tokens,
                         "temperature": p.temperature, "token_target": p.token_target, "token_limit": p.token_limit,
                         "output": p.output, "layout": p.layout}
                        for p in presets.values()],
            "modifiers": [{"id": m.id, "name": m.name, "text": m.text, "takes_value": m.takes_value}
                          for m in mods.values()],
            "queue": _read_json(QUEUE_PATH, {"items": []}),
            "job": JOB.poll(0)["state"],
        })

    def models(self, q):
        s = config.load_settings()
        favs = s.get("favorite_models", [])
        client = make_client(s)
        models = client.list_models(include_all=q.get("all") == "1")
        rows = [{"id": m.id, "display": m.display, "type": m.type, "vision": m.vision, "loaded": m.loaded,
                 "quant": m.quantization, "fav": m.id in favs, "max_ctx": m.max_context,
                 "size": f"{m.size_bytes / 1e9:.1f} GB" if m.size_bytes else ""} for m in models]
        rows.sort(key=lambda r: (not r["fav"], not r["loaded"], r["id"].lower()))
        loaded = [{"id": m.id, "identifier": m.identifier, "display": m.display, "ctx": m.loaded_context,
                   "vision": m.vision} for m in client.loaded_models()]
        self._json({"models": rows, "loaded": loaded, "last": s.get("last_model", "")})

    def modes(self, q):
        s = config.load_settings()
        self._json(model_modes(make_client(s), q.get("model") or USE_LOADED))

    def load_model(self, body):
        s = config.load_settings()
        if JOB.poll(0)["state"] == "running":
            return self._error("a run is in progress - stop it first")
        model = body.get("model", "")
        if not model or model == USE_LOADED:
            return self._error("pick a model to load")
        secs = make_client(s).ensure_loaded(model, int(body.get("context") or s["load_context_length"]), True)
        self._json({"ok": True, "load_s": round(secs, 1)})

    def eject_model(self, body):
        s = config.load_settings()
        if JOB.poll(0)["state"] == "running":
            return self._error("a run is in progress - stop it first")
        client = make_client(s)
        for m in client.loaded_models():
            if not body.get("model") or body["model"] in (m.id, m.identifier):
                client.unload(m.identifier or m.id)
        self._json({"ok": True})

    def favorite(self, body):
        with SETTINGS_LOCK:
            s = config.load_settings()
            favs = [m for m in s.get("favorite_models", []) if m != body["model"]]
            if body.get("on"):
                favs.append(body["model"])
            s["favorite_models"] = favs
            config.save_settings(s)
        self._json({"ok": True, "favorites": favs})

    def settings_put(self, body):
        allowed = {"group_dir", "load_context_length", "max_side", "default_preset", "wildcards_dir", "sweeps_dir",
                   "lmstudio_models_dir", "comfyui_url", "render_gpu", "render_free_after", "render_preset",
                   "wc_nsfw", "wc_violence", "wc_content_filter", "wc_strip_lora", "lora_catalog_url", "lora_catalog_db", "sweeps_root"}
        with SETTINGS_LOCK:
            s = config.load_settings()
            for k, v in body.items():
                if k in allowed:
                    s[k] = v
            config.save_settings(s)
        self._json({"ok": True})

    # --- browsing
    def drives(self, q):
        s = config.load_settings()
        self._json({"drives": _drives(), "recent": s.get("recent_folders", [])})

    def ls(self, q):
        s = config.load_settings()
        result = _ls(q.get("path") or _drives()[0], s["per_image_suffix"], s["per_image_suffix_json"])
        if not result.get("error") and result["images"]:
            with SETTINGS_LOCK:
                s = config.load_settings()
                _remember_folder(s, result["path"])
                config.save_settings(s)
        self._json(result)

    def media(self, q):
        """The original file, with byte ranges so the browser can seek in videos."""
        import mimetypes
        path = q.get("path", "")
        if not (os.path.isfile(path) and (is_media(path) or path.lower().endswith(".svg"))):
            return self._error("not found", 404)
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range") or ""
        if rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a.strip():
                start = int(a)
                end = min(int(b), size - 1) if b.strip() else size - 1
            elif b.strip():
                start = max(0, size - int(b))
            status = 206
        length = max(0, end - start + 1)
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                left = length
                while left > 0:
                    chunk = handle.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionError, BrokenPipeError):
            pass

    def thumb(self, q):
        path = q.get("path", "")
        if path.lower().endswith(".svg") and os.path.isfile(path):   # vector renders: the browser draws them itself
            with open(path, "rb") as handle:
                return self._send(200, handle.read(), "image/svg+xml")
        if not (os.path.isfile(path) and is_media(path)):
            return self._error("not an image", 404)
        side = max(64, min(1600, int(q.get("side", 256))))
        self._send(200, _thumb(path, side), "image/jpeg")

    def expand(self, body):
        found = discover(body.get("paths", []), recursive=bool(body.get("recursive", True)))
        self._json({"items": [{"image": img, "root": root} for img, root in found]})

    def upload(self):
        n = int(self.headers.get("Content-Length") or 0)
        rel = self.headers.get("X-Filename", "upload.png")
        batch = _SAFE_NAME.sub("_", self.headers.get("X-Batch", "") or time.strftime("%Y%m%d_%H%M%S"))
        parts = [(_SAFE_NAME.sub("_", p) or "_") for p in re.split(r"[\\/]+", rel) if p not in ("", ".", "..")]
        if not parts or not is_media(parts[-1]):
            self.rfile.read(n)
            return self._error("not an image file")
        root = os.path.join(STAGING_DIR, batch)
        target = os.path.join(root, *parts)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            remaining = n
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        self._json({"image": target, "root": root})

    # --- queue
    def queue_get(self, q):
        self._json(_read_json(QUEUE_PATH, {"items": []}))

    def queue_put(self, body):
        _write_json(QUEUE_PATH, {"items": body.get("items", []), "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._json({"ok": True})

    def queues(self, q):
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(QUEUES_DIR)) if os.path.isdir(QUEUES_DIR) else []
        self._json({"names": names})

    def queues_save(self, body):
        name = _SAFE_NAME.sub("_", body.get("name", "").strip())
        if not name:
            return self._error("name required")
        _write_json(os.path.join(QUEUES_DIR, name + ".json"), {"items": body.get("items", [])})
        self._json({"ok": True, "name": name})

    def queues_load(self, q):
        name = _SAFE_NAME.sub("_", q.get("name", ""))
        self._json(_read_json(os.path.join(QUEUES_DIR, name + ".json"), {"items": []}))

    # --- instructions
    def preset_get(self, q):
        pid = q.get("id", "")
        path = os.path.join(config.PRESETS_DIR, pid + ".yaml")
        if not os.path.isfile(path):
            return self._error("unknown preset", 404)
        with open(path, "r", encoding="utf-8") as handle:
            self._json({"id": pid, "text": handle.read()})

    def preset_save(self, body):
        pid = _SAFE_NAME.sub("_", body.get("id", "").strip()).replace(" ", "_")
        text = body.get("text", "")
        if not pid:
            return self._error("preset id required")
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            return self._error(f"YAML error: {exc}")
        if not isinstance(data, dict) or not data.get("system"):
            return self._error("a preset needs at least a 'system' field")
        if data.get("id") != pid:                      # the file name is the id; keep them in step
            text = re.sub(r"(?m)^id:.*$", f"id: {pid}", text) if re.search(r"(?m)^id:", text) else f"id: {pid}\n{text}"
        path = os.path.join(config.PRESETS_DIR, pid + ".yaml")
        if os.path.isfile(path):
            bak = os.path.join(config.PRESETS_DIR, ".bak")
            os.makedirs(bak, exist_ok=True)
            shutil.copy2(path, os.path.join(bak, f"{pid}_{time.strftime('%Y%m%d_%H%M%S')}.yaml"))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        try:                                            # prove it composes
            compose(load_presets(config.PRESETS_DIR)[pid])
        except Exception as exc:
            return self._json({"ok": True, "id": pid, "warning": f"saved, but it does not compose: {exc}"})
        self._json({"ok": True, "id": pid})

    def compose_preview(self, body):
        presets = load_presets(config.PRESETS_DIR)
        preset = presets[body.get("preset") or config.load_settings()["default_preset"]]
        mods = parse_modifier_args(body.get("mods", []), load_modifiers(config.MODIFIERS_DIR))
        c = compose(preset, mods, body.get("note", ""), int(body.get("variations") or 1))
        self._json({"system": c.system, "user": c.user, "sha": c.sha})

    # --- generation
    def run(self, body):
        from . import vision
        if vision.JOB.status().get("running"):
            return self._error("a vision check run is using LM Studio - wait for it or stop it in the Review tab")
        items = _items(body)
        if not items:
            return self._error("queue is empty")
        if not body.get("model"):
            return self._error("pick a model")
        s = config.load_settings()

        def opt_float(k):
            v = body.get(k)
            return None if v in (None, "") else float(v)

        spec = JobSpec(items=items, model=body["model"], preset=body.get("preset", ""), mods=body.get("mods", []),
                       note=body.get("note", ""), notes=body.get("notes", {}) or {},
                       variations=int(body.get("variations") or 1), reasoning=body.get("reasoning", ""),
                       max_tokens=int(body.get("max_tokens") or 0), temperature=opt_float("temperature"),
                       seed=(int(body["seed"]) if body.get("seed") not in (None, "") else None),
                       force=bool(body.get("force")), force_images=body.get("redo", []) or [],
                       context_length=int(body.get("context_length") or 0), facts=bool(body.get("facts")),
                       facts_ethnicity=bool(body.get("facts_ethnicity")), layout=body.get("layout") or {}, trigger=body.get("trigger", ""), argv=["gui"])
        with SETTINGS_LOCK:
            s["last_model"] = body["model"]
            config.save_settings(s)
        JOB.start(s, spec)
        self._json({"ok": True})

    def job(self, q):
        self._json(JOB.poll(int(q.get("since", 0))))

    def job_cancel(self, body):
        JOB.cancel.set()
        self._json({"ok": True})

    # --- outputs
    def plan(self, body):
        s = config.load_settings()
        p = plan_outputs(s, _items(body), body.get("group_by", "folder"), body.get("overrides") or {})

        def existing(name):                         # Nov 14: the Save tab says "new file" / "adds to N lines"
            path = name if os.path.isabs(name) else os.path.join(s["group_dir"], name)
            if not os.path.isfile(path):
                return None
            try:
                with open(path, "rb") as handle:
                    return sum(1 for line in handle if line.strip())
            except OSError:
                return 0
        for g in p["groups"]:
            g["existing_lines"] = existing(g["name"])
        self._json({"groups": p["groups"], "combined": p["combined"], "group_dir": s["group_dir"],
                    "combined_existing_lines": existing(p["combined"])})

    def lists(self, q):
        s = config.load_settings()
        folder = q.get("dir") or s["group_dir"]
        try:
            names = sorted((e.name for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith((".txt", ".jsonl"))),
                           key=natural_key)
        except OSError:
            names = []
        self._json({"dir": folder, "names": names})

    def export(self, body):
        s = config.load_settings()
        items = _items(body)
        prompts = {it["image"]: it.get("prompts") or [] for it in body.get("items", [])}
        if body.get("modes") == ["caption"]:                  # training captions: first prompt -> <stem>.txt
            from .outputs import write_caption
            from .store import append_index
            written = [write_caption(img, ps[0]) for img, ps in prompts.items() if ps and ps[0].strip()]
            append_index(config.RUNS_DIR, [{"file": w["path"], "mode": "caption", "image": img, "prompt": ps[0]}
                                           for (img, ps), w in zip([(i, p) for i, p in prompts.items() if p and p[0].strip()], written)
                                           if w["action"] == "new"])
            return self._json({"written": written})
        overrides = body.get("overrides") or {}
        written = write_outputs(s, items, prompts, body.get("modes", []), body.get("format", "lines"),
                                body.get("group_by", "folder"), body.get("group_dir", "") or s["group_dir"],
                                body.get("combined", ""), overrides, body.get("on_exists", "new"),
                                bool(body.get("dedupe")),
                                {it["image"]: it["meta"] for it in body.get("items", []) if it.get("meta")},
                                bool(body.get("sidecar")))
        if overrides and body.get("remember", True):
            with SETTINGS_LOCK:
                s = config.load_settings()
                s["group_name_overrides"] = {**s.get("group_name_overrides", {}), **overrides}
                config.save_settings(s)
        self._json({"written": written})

    # --- Prompt Review (phase 1) -------------------------------------------------------------------------------
    def review_sources(self, q):
        from . import review
        s = config.load_settings()
        folders = []
        counts = review.rated_counts(s.get("review_folders", []))
        for f in s.get("review_folders", []):
            folders.append({"kind": "folder", "path": f, "name": os.path.basename(f.rstrip("\\/")) or f,
                            "exists": os.path.isdir(f), "rated": counts.get(os.path.normcase(f), 0)})
        self._json({"sweeps": review.list_sweeps(s.get("sweeps_dir", "")), "folders": folders,
                    "sweeps_dir": s.get("sweeps_dir", ""),
                    "defects": [{"id": d[0], "key": d[1], "label": d[2]} for d in review.DEFECTS]})

    def review_folders(self, body):
        with SETTINGS_LOCK:
            s = config.load_settings()
            folders = [f for f in s.get("review_folders", [])]
            if body.get("add"):
                path = os.path.abspath(body["add"].strip().strip('"'))
                if not os.path.isdir(path):
                    return self._error(f"not a folder: {path}")
                if os.path.normcase(path) not in [os.path.normcase(f) for f in folders]:
                    folders.insert(0, path)
            if body.get("remove"):
                folders = [f for f in folders if os.path.normcase(f) != os.path.normcase(body["remove"])]
            s["review_folders"] = folders
            config.save_settings(s)
        self._json({"ok": True, "folders": folders})

    def review_load(self, q):
        from . import review
        path = q.get("path", "")
        if q.get("kind") == "sweep":
            data = review.load_sweep(path)
        else:
            if not os.path.isdir(path):
                return self._error(f"not a folder: {path}")
            data = review.load_folder(path, recursive=q.get("recursive", "1") == "1")
        self._json(data)

    def review_rate(self, body):
        from . import review
        fields = {k: body[k] for k in ("stars", "keeper", "defects", "adherence", "fidelity", "note", "lesson")
                  if k in body}
        rating = review.set_rating(body["file"], body.get("source", ""), body.get("sha", ""), fields)
        self._json({"ok": True, "rating": rating})

    def review_export(self, body):
        from . import review
        path = body.get("path", "")
        data = review.load_sweep(path) if body.get("kind") == "sweep" else review.load_folder(path, body.get("recursive", True))
        self._json({"path": review.export_csv(data)})

    # --- Reports (phase 3) ------------------------------------------------------------------------------
    def report_page(self, q):
        with open(os.path.join(HERE, "report.html"), "rb") as handle:
            self._send(200, handle.read(), "text/html; charset=utf-8")

    def review_report(self, q):
        from . import report
        self._json(report.build(q.get("kind", "sweep"), q.get("path", "")))

    # --- Settings page + Render (ComfyUI) ---------------------------------------------------------------------
    # --- prompt grades + library ----------------------------------------------------------------------------------
    def pgrade(self, body):
        from . import prompt_grades
        self._json(prompt_grades.set_grade(body.get("prompt", ""), body.get("grade"), body.get("reasons"),
                                           body.get("tags"), body.get("note"), body.get("source", ""),
                                           body.get("provenance") or {}, body.get("reference", "")))

    def pgrades(self, body):
        from . import prompt_grades
        from .outputs import prompt_sha
        shas = list(body.get("shas") or []) + [prompt_sha(" ".join((t or "").split())) for t in body.get("texts") or []]
        self._json({"grades": prompt_grades.grades_for(shas)})

    def library_api(self, q):
        from . import prompt_grades
        self._json(prompt_grades.library(q.get("grade", "great"), q.get("q", ""), q.get("tag", "")))

    # -- phase 2 (Nov 20 2026): lessons + suggested grades ----------------------------------------------------------
    def lessons_list(self, q):
        from . import prompt_lessons
        self._json({"lessons": prompt_lessons.lessons()})

    def lessons_propose(self, body):
        from . import prompt_lessons
        try:
            self._json(prompt_lessons.propose(config.load_settings(), body.get("model") or "__loaded__"))
        except Exception as e:                               # LM Studio not reachable / nothing loaded
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def lessons_status(self, body):
        from . import prompt_lessons
        self._json(prompt_lessons.set_status(int(body.get("id") or 0), body.get("status", ""), body.get("rule")))

    def grade_suggestions(self, q):
        from . import prompt_lessons
        self._json(prompt_lessons.suggestions(config.load_settings()))

    def library_forensics(self, q):
        from . import prompt_grades
        self._json(prompt_grades.forensics())

    def library_export(self, body):
        from . import lists, prompt_grades
        rows = [r for r in prompt_grades.library(body.get("grade", "great"))["rows"]
                if not body.get("shas") or r["sha"] in set(body["shas"])]
        entries = [{"text": r["prompt"], "header": ", ".join(r["tags"]) or r["grade"],
                    "meta": {"grade": r["grade"], "tags": r["tags"], "reasons": r["reasons"], **r["provenance"]},
                    "index": {"mode": "library", "image": r["provenance"].get("source_image", ""), "variant": "library"}}
                   for r in rows]
        if body.get("grade") == "duff":                       # the archive can be exported for forensics on purpose
            from . import outputs
            import os
            s = config.load_settings()
            path = outputs.unique_path(os.path.join(s["group_dir"], (body.get("name") or "duff_archive") + ".txt"))
            with open(path, "w", encoding="utf-8", newline="\n") as h:
                h.write("\n".join(e["text"] for e in entries) + "\n")
            return self._json({"path": path, "file": os.path.basename(path), "added": len(entries), "skipped": 0,
                               "skipped_duff": 0, "action": "new"})
        self._json(lists.save_list(config.load_settings(), entries, body.get("name", ""), "library", **_list_opts(body)))

    def asset(self, q):
        """Images from the app's assets folder (the splash uses assets/hero.jpg) - file names only, images only."""
        name = os.path.basename(q.get("name", ""))
        ext = os.path.splitext(name)[1].lower()
        types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        path = os.path.join(config.APP_DIR, "assets", name)
        if ext not in types or not os.path.isfile(path):
            self.send_response(404)
            self.end_headers()
            return
        with open(path, "rb") as handle:
            data = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", types[ext])
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")            # swapped pictures show at once
        self.end_headers()
        self.wfile.write(data)

    def library_page(self, q):
        with open(os.path.join(HERE, "library.html"), "rb") as handle:
            data = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def wc_search(self, q):
        from . import wc
        try:
            self._json(wc.search(config.load_settings(), q.get("q", ""), q.get("facet", ""), int(q.get("limit") or 80),
                                 small=q.get("small") == "1"))
        except ValueError as exc:
            self._json({"rows": [], "facets": [], "total": 0, "error": str(exc)})

    def wc_entries(self, q):
        from . import wc
        self._json(wc.entries(config.load_settings(), q.get("key", ""), int(q.get("limit") or 400)))

    def wc_roll(self, body):
        from . import wc
        seed = body.get("seed")
        self._json(wc.roll(config.load_settings(), body.get("template", ""), int(body.get("n") or 5),
                           int(seed) if str(seed or "").lstrip("-").isdigit() else None))

    def lists_peek(self, q):
        from . import lists
        self._json(lists.peek(config.load_settings(), q.get("name", ""), q.get("stem", "list"), q.get("fmt", "lines")))

    def settings_all(self, q):
        s = config.load_settings()
        keys = ("group_dir", "wildcards_dir", "sweeps_dir", "lmstudio_models_dir", "load_context_length", "comfyui_url",
                "render_gpu", "render_free_after", "render_preset", "wc_nsfw", "wc_violence", "wc_content_filter",
                "wc_strip_lora", "lora_catalog_url", "lora_catalog_db", "sweeps_root")
        self._json({k: s.get(k) for k in keys})

    def r_state(self, q):
        from . import render
        s = config.load_settings()
        self._json({"presets": render.list_presets(), "workflows": render.list_workflows(),
                    "current": s.get("render_preset", ""), "gpu": s.get("render_gpu", "shared"),
                    "job": render.JOB.status(10 ** 9)})

    def r_status(self, q):
        from . import render
        self._json(render.JOB.status(int(q.get("since") or 0)))

    def r_fields(self, q):
        from . import render
        wf = render.load_workflow(q.get("name", ""))
        self._json({"fields": render.fields(wf), "mapping": render.load_mapping(q.get("name", ""))})

    def r_for(self, q):
        from . import render
        self._json(render.renders_for([q.get("key", "")]))

    def r_for_many(self, body):
        from . import render
        texts = [" ".join((t or "").split()) for t in body.get("texts") or []]
        keys = {render.prompt_key(t): t for t in texts if t}
        found = render.renders_for(list(keys))
        self._json({"by_text": {keys[k]: v for k, v in found.items()}})

    def r_fromlast(self, body):
        from . import render
        self._json(render.save_last_run(body.get("name", "")))

    def r_map(self, body):
        from . import render
        render.save_mapping(body["name"], body.get("mapping") or {})
        self._json({"ok": True, "mapping": render.load_mapping(body["name"])})

    # --- saved Photobash pools ---------------------------------------------------------------------------------
    def pools_list(self, q):
        from . import pools
        self._json({"pools": pools.list_pools(q.get("q", ""))})

    def pools_load(self, q):
        from . import pools
        self._json(pools.load_pool(q.get("id", "")))

    def pools_save(self, body):
        from . import pools
        self._json(pools.save_pool(body.get("name", ""), body.get("description", ""), body.get("files") or [],
                                   body.get("folders") or [], body.get("id") or None))

    def pools_delete(self, body):
        from . import pools
        self._json(pools.delete_pool(body.get("id", "")))

    def r_apikey(self, body):
        from .image_apis import PROVIDERS
        b = body.get("backend", "")
        if b not in PROVIDERS:
            return self._error("unknown image API")
        with SETTINGS_LOCK:
            s = config.load_settings()
            keys = dict(s.get("api_keys") or {})
            k = (body.get("key") or "").strip()
            if k:
                keys[b] = k
            else:
                keys.pop(b, None)
            s["api_keys"] = keys
            config.save_settings(s)
        self._json({"backend": b, "has_key": bool(keys.get(b))})

    def layout_from_image(self, body):
        """Describe one image (usually a render) as an Ideogram 4 JSON prompt - boxes measured from the picture."""
        from .generate import GenParams, generate_one
        from .instructions import compose, load_presets
        from .jobs import make_client, resolve_model
        from .model_caps import caps_for
        path = body.get("path", "")
        if not os.path.isfile(path):
            return self._error("image not found")
        if self._lm_busy():
            return self._error("LM Studio is busy with another run - wait for it first")
        preset = load_presets(config.PRESETS_DIR)["ideogram4"]
        client = make_client(config.load_settings())
        try:
            api_id, gguf, _ = resolve_model(client, "__loaded__")
        except Exception as exc:
            return self._error(f"load a vision model in LM Studio first ({exc})")
        r = generate_one(client, path, preset, compose(preset, [], "", 1), GenParams(model=api_id, caps=caps_for(gguf)))
        if r.error or not r.prompts:
            return self._error(r.error or "the model returned nothing")
        self._json({"prompt": r.prompts[0], "model": api_id})

    def r_caps(self, q):
        from . import image_apis
        b = q.get("backend", "")
        key = (config.load_settings().get("api_keys") or {}).get(b, "")
        try:
            self._json({"caps": image_apis.model_caps(b, q.get("model", "").strip(), key)})
        except image_apis.ApiError as e:
            self._error(str(e))

    def r_providers(self, q):
        from .image_apis import PROVIDERS
        keys = config.load_settings().get("api_keys") or {}
        self._json({"providers": {k: {**v, "has_key": bool(keys.get(k))} for k, v in PROVIDERS.items()}})

    def r_preset(self, body):
        from . import render
        self._json(render.save_preset(body))

    def r_start(self, body):
        from . import render
        p = body.get("preset") or ""
        if p:
            with SETTINGS_LOCK:
                s = config.load_settings()
                s["render_preset"] = p
                config.save_settings(s)
        preset = render.get_preset(p) if p else {}
        api = preset.get("backend", "comfyui") != "comfyui"
        if api and not body.get("confirm_cost"):          # hosted APIs cost money: the page must confirm first
            from .image_apis import PROVIDERS
            n = len([i for i in body.get("items") or [] if (i.get("text") or "").strip()]) * int(preset.get("batch") or 1)
            price = float(preset.get("price") or 0)
            return self._json({"needs_confirm": True, "images": n, "service": PROVIDERS[preset["backend"]]["label"],
                               "model": preset.get("model", ""), "price": price, "estimate": round(n * price, 2)})
        if not api and config.load_settings().get("render_gpu", "shared") == "shared" and self._lm_busy():
            return self._error("a prompt job is using LM Studio - on one GPU wait for it, or set GPU sharing to separate")
        self._json(render.JOB.start(body.get("items") or [], p))

    def r_stop(self, body):
        from . import render
        render.JOB.stop()
        self._json({"ok": True})

    # --- Rewrite ------------------------------------------------------------------------------------------
    def _lm_busy(self):
        from . import audit, compare, ideas, photobash, vision
        return (JOB.poll(0)["state"] == "running" or vision.JOB.status().get("running") or audit.JOB.status().get("running")
                or compare.JOB.status().get("running") or photobash.JOB.status().get("running")
                or ideas.JOB.status().get("running"))

    def rw_start(self, body):
        from . import rewrite
        if self._lm_busy():
            return self._error("LM Studio is busy with another run - wait for it first")
        self._json(rewrite.JOB.start(config.load_settings(), body))

    def rw_status(self, q):
        from . import rewrite
        self._json(rewrite.JOB.status(int(q.get("since") or 0)))

    def rw_stop(self, body):
        from . import rewrite
        rewrite.JOB.stop()
        self._json({"ok": True})

    def rw_save(self, body):
        from . import rewrite
        rows = body.get("results") or []
        if rows and all(r.get("op") == "pair" for r in rows):
            return self._json({"path": rewrite.save_pairs(config.load_settings(), rows, body.get("name", ""))})
        from . import lists
        self._json(lists.save_list(config.load_settings(), lists.rewrite_entries(rows), body.get("name", ""), "rewrite",
                                   **_list_opts(body)))

    def rw_load(self, body):
        from . import rewrite
        p = (body.get("path") or "").strip().strip('"')
        if not os.path.isfile(p):
            return self._error(f"not a file: {p}")
        self._json({"items": rewrite.load_list(p)})

    def rw_split(self, body):
        from . import rewrite
        self._json({"items": [{"text": t, "image": ""} for t in rewrite.split_text(body.get("text", ""))]})

    # --- Ideas (no images) -------------------------------------------------------------------------------
    def id_start(self, body):
        from . import audit, compare, ideas, photobash, vision
        needs = body.get("mode") != "wildcards" or body.get("polish")
        if needs and (JOB.poll(0)["state"] == "running" or vision.JOB.status().get("running")
                      or audit.JOB.status().get("running") or compare.JOB.status().get("running")
                      or photobash.JOB.status().get("running")):
            return self._error("LM Studio is busy with another run - wait for it first")
        self._json(ideas.JOB.start(config.load_settings(), body))

    def id_status(self, q):
        from . import ideas
        self._json(ideas.JOB.status(int(q.get("since") or 0)))

    def id_stop(self, body):
        from . import ideas
        ideas.JOB.stop()
        self._json({"ok": True})

    def id_save(self, body):
        from . import ideas
        from . import lists
        self._json(lists.save_list(config.load_settings(), lists.ideas_entries(body.get("results") or []),
                                   body.get("name", ""), "ideas", **_list_opts(body)))

    def id_wildcards(self, q):
        from . import ideas
        s = config.load_settings()
        try:
            wc = ideas.wildcards(s.get("wildcards_dir") or "")
        except ValueError as exc:
            return self._json({"hits": [], "total": 0, "error": str(exc)})
        self._json({"hits": wc.search(q.get("q", "")), "total": len(wc.by_rel)})

    # --- Photobash ---------------------------------------------------------------------------------------
    def pb_start(self, body):
        from . import audit, compare, photobash, vision
        if (JOB.poll(0)["state"] == "running" or vision.JOB.status().get("running") or audit.JOB.status().get("running")
                or compare.JOB.status().get("running")):
            return self._error("LM Studio is busy with another run - wait for it first")
        self._json(photobash.JOB.start(config.load_settings(), body))

    def pb_preview(self, body):
        from . import photobash
        self._json(photobash.preview(config.load_settings(), body))

    def pb_status(self, q):
        from . import photobash
        self._json(photobash.JOB.status(int(q.get("since") or 0)))

    def pb_stop(self, body):
        from . import photobash
        photobash.JOB.stop()
        self._json({"ok": True})

    def pb_save(self, body):
        from . import photobash
        from . import lists
        self._json(lists.save_list(config.load_settings(), lists.photobash_entries(body.get("results") or []),
                                   body.get("name", ""), "photobash", **_list_opts(body)))

    # --- Compare runs: several prompt versions per image ------------------------------------------------
    def compare_start(self, body):
        from . import audit, compare, vision
        if JOB.poll(0)["state"] == "running" or vision.JOB.status().get("running") or audit.JOB.status().get("running"):
            return self._error("LM Studio is busy with another run - wait for it first")
        items = _items(body)
        base = {"model": body.get("model", ""), "mods": body.get("mods") or [], "note": body.get("note", ""),
                "max_tokens": body.get("max_tokens") or 0, "temperature": body.get("temperature"),
                "seed": body.get("seed"), "layout": body.get("layout") or {}, "trigger": body.get("trigger", "")}
        self._json(compare.JOB.start(config.load_settings(), items, body.get("variants") or [], base,
                                     body.get("name", "")))

    def compare_stop(self, body):
        from . import compare
        compare.JOB.stop()
        self._json({"ok": True})

    def compare_status(self, q):
        from . import compare
        self._json(compare.JOB.status())

    # --- Audit: vision model vs detectors ----------------------------------------------------------------
    def audit_page(self, q):
        with open(os.path.join(HERE, "audit.html"), "rb") as handle:
            self._send(200, handle.read(), "text/html; charset=utf-8")

    def audit_results(self, q):
        from . import audit
        self._json(audit.results(q.get("batch") or None))

    def audit_status(self, q):
        from . import audit
        self._json(audit.JOB.status())

    def audit_start(self, body):
        from . import audit, vision
        if JOB.poll(0)["state"] == "running" or vision.JOB.status().get("running"):
            return self._error("LM Studio is busy with a generation run or vision checks - wait for it first")
        folder = (body.get("folder") or "").strip().strip('"')
        if folder and not os.path.isdir(folder):
            return self._error(f"not a folder: {folder}")
        self._json(audit.JOB.start(config.load_settings(), int(body.get("limit") or 100), body.get("model", ""),
                                   bool(body.get("redo")), folder))

    def audit_stop(self, body):
        from . import audit
        audit.JOB.stop()
        self._json({"ok": True})

    def audit_verdict(self, body):
        from . import audit
        audit.set_verdict(body["image"], body["field"], body.get("verdict", ""))
        self._json({"ok": True})

    def check_start(self, body):
        from . import vision
        if JOB.poll(0)["state"] == "running":
            return self._error("a prompt-generation run is using LM Studio - wait for it or stop it first")
        st = vision.JOB.start(config.load_settings(), body.get("kind", "sweep"), body["path"], body.get("model", ""),
                              body.get("mode", "off"), bool(body.get("force")), bool(body.get("fidelity", True)),
                              body["only"] if "only" in body else None)          # [] = nothing, absent = all
        self._json(st)

    def check_stop(self, body):
        from . import vision
        vision.JOB.stop()
        self._json({"ok": True})

    def check_status(self, q):
        from . import vision
        self._json(vision.JOB.status())

    def check_detail(self, q):
        from . import vision
        self._json({"detail": vision.check_detail(q.get("file", ""))})

    def whence(self, body):
        from .store import search_index
        self._json({"hits": search_index(config.RUNS_DIR, body.get("text", ""))})

    def open_path(self, body):
        path = os.path.abspath(body.get("path", ""))
        if os.path.isfile(path):
            subprocess.Popen(["explorer", "/select,", path])
        elif os.path.isdir(path):
            os.startfile(path)
        else:
            return self._error("not found", 404)
        self._json({"ok": True})


def serve(port: int = 8790, open_browser: bool = True) -> None:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"VLM Prompter GUI at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
