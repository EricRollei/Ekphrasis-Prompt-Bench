"""
Render prompts with ComfyUI (phase 1 of private_docs/SPEC_render.md).

  workflows/<name>.json          ComfyUI API-format workflow (saved from ComfyUI's history: "use my last run")
  workflows/<name>.map.json      which node fields take prompt / negative / seed / width / height
  render_presets/<name>.json     backend + workflow + batch + seed mode + size
  renders/<date>/...png          rendered images (copies fetched from ComfyUI), index in renders/index.jsonl

One GPU: before rendering, the LM Studio model is unloaded (remembered) and reloaded when the next prompt job needs
it (jobs.resolve_model -> reload_swapped); after a render batch ComfyUI can be asked to free its VRAM.
All three folders are git-ignored - they hold the user's own models and paths.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import random
import re
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from . import config

WF_DIR = os.path.join(config.APP_DIR, "workflows")
PRESET_DIR = os.path.join(config.APP_DIR, "render_presets")
RENDER_DIR = os.path.join(config.APP_DIR, "renders")
INDEX = os.path.join(RENDER_DIR, "index.jsonl")
ROLES = ["prompt", "negative", "seed", "width", "height"]
_NAME = re.compile(r"[^A-Za-z0-9_.+ -]+")


def safe_name(name: str) -> str:
    n = _NAME.sub("_", (name or "").strip()).strip(" .")
    if not n:
        raise ValueError("give it a name")
    return n[:80]


def prompt_key(text: str) -> str:
    return hashlib.sha1(" ".join((text or "").split()).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- ComfyUI client
class Comfy:
    def __init__(self, url: str):
        self.url = (url or "http://127.0.0.1:8188").rstrip("/")

    def _req(self, path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 30, raw: bool = False):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI {path}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"ComfyUI not reachable at {self.url} ({e.reason}) - is it running?")
        return payload if raw else (json.loads(payload) if payload else {})

    def last_run(self) -> Dict[str, Any]:
        h = self._req("/history?max_items=1")
        if not h:
            raise RuntimeError("ComfyUI's history is empty - run your workflow once in ComfyUI first")
        entry = list(h.values())[-1]
        return entry["prompt"][2]

    def queue(self, wf: Dict[str, Any]) -> str:
        r = self._req("/prompt", {"prompt": wf, "client_id": "ekphrasis-prompt-bench"})
        if r.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected the workflow: {json.dumps(r['node_errors'])[:400]}")
        return r["prompt_id"]

    def wait(self, prompt_id: str, cancel: threading.Event, timeout_s: float = 3600) -> Dict[str, Any]:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if cancel.is_set():
                try:
                    self._req("/interrupt", {})
                except RuntimeError:
                    pass
                raise InterruptedError
            h = self._req(f"/history/{prompt_id}")
            if prompt_id in h:
                entry = h[prompt_id]
                st = entry.get("status") or {}
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                    raise RuntimeError("ComfyUI error: " + (json.dumps(msgs[-1][1])[:400] if msgs else "unknown"))
                return entry
            time.sleep(1.0)
        raise TimeoutError("render took too long")

    def image(self, filename: str, subfolder: str, kind: str) -> bytes:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": kind})
        return self._req("/view?" + q, raw=True, timeout=120)

    def free(self) -> None:
        self._req("/free", {"unload_models": True, "free_memory": True})


def comfy() -> Comfy:
    return Comfy(config.load_settings().get("comfyui_url"))


# ----------------------------------------------------------------------------------------------- workflows
def list_workflows() -> List[Dict[str, Any]]:
    os.makedirs(WF_DIR, exist_ok=True)
    out = []
    for n in sorted(os.listdir(WF_DIR)):
        if n.endswith(".json") and not n.endswith(".map.json"):
            name = n[:-5]
            out.append({"name": name, "mapped": os.path.isfile(os.path.join(WF_DIR, name + ".map.json"))})
    return out


def load_workflow(name: str) -> Dict[str, Any]:
    with open(os.path.join(WF_DIR, safe_name(name) + ".json"), "r", encoding="utf-8") as h:
        return json.load(h)


def save_last_run(name: str) -> Dict[str, Any]:
    name = safe_name(name)
    wf = comfy().last_run()
    os.makedirs(WF_DIR, exist_ok=True)
    with open(os.path.join(WF_DIR, name + ".json"), "w", encoding="utf-8") as h:
        json.dump(wf, h, indent=1, ensure_ascii=False)
    mp = os.path.join(WF_DIR, name + ".map.json")
    if not os.path.isfile(mp):
        save_mapping(name, guess_mapping(wf))
    return {"name": name, "nodes": len(wf), "mapping": load_mapping(name)}


def fields(wf: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Plain (non-linked) inputs a role can point at."""
    out = []
    for nid, node in wf.items():
        title = (node.get("_meta") or {}).get("title", "")
        for k, v in (node.get("inputs") or {}).items():
            if isinstance(v, list):                         # a link to another node
                continue
            kind = "text" if isinstance(v, str) else "bool" if isinstance(v, bool) else "number"
            if kind == "bool":
                continue
            out.append({"node": nid, "input": k, "title": title, "class": node.get("class_type", ""), "kind": kind,
                        "value": v if kind == "number" else str(v)[:120]})
    return out


def guess_mapping(wf: Dict[str, Any]) -> Dict[str, Any]:
    fs = fields(wf)

    def pick(pred):
        for f in fs:
            if pred(f):
                return [f["node"], f["input"]]
        return None

    texts = [f for f in fs if f["kind"] == "text"]
    m = {
        "prompt": pick(lambda f: f["kind"] == "text" and "prompt" in f["title"].lower() and "neg" not in f["title"].lower())
        or pick(lambda f: f["kind"] == "text" and f["input"] in ("prompt", "positive", "text") and "neg" not in f["title"].lower()
                and f["class"] in ("Text Multiline", "CLIPTextEncode", "PrimitiveString", "PrimitiveStringMultiline"))
        or (([texts[0]["node"], texts[0]["input"]]) if len(texts) == 1 else None),
        "negative": pick(lambda f: f["kind"] == "text" and f["input"] in ("negative_prompt", "negative")),
        "seed": pick(lambda f: f["kind"] == "number" and f["input"] in ("seed", "noise_seed")),
        "width": pick(lambda f: f["kind"] == "number" and f["input"] == "width"),
        "height": pick(lambda f: f["kind"] == "number" and f["input"] == "height"),
    }
    return {k: v for k, v in m.items() if v}


def load_mapping(name: str) -> Dict[str, Any]:
    p = os.path.join(WF_DIR, safe_name(name) + ".map.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as h:
        return json.load(h)


def save_mapping(name: str, mapping: Dict[str, Any]) -> None:
    clean = {k: v for k, v in (mapping or {}).items() if k in ROLES and isinstance(v, list) and len(v) == 2}
    with open(os.path.join(WF_DIR, safe_name(name) + ".map.json"), "w", encoding="utf-8") as h:
        json.dump(clean, h, indent=1)


_UNLOAD_HINTS = ("unload", "freememory", "free_memory", "cleanvram", "clean_vram", "purgevram", "purge_vram")


def unload_nodes(wf: Dict[str, Any]) -> List[str]:
    """Nodes that unload models / free VRAM (by class name), e.g. EricKrea2UnloadModels."""
    return [nid for nid, n in wf.items() if any(h in n.get("class_type", "").lower().replace(" ", "") for h in _UNLOAD_HINTS)]


def without_unload(wf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The workflow minus its unload node(s) and anything that only hangs off them (display nodes ...).
    None when there is nothing to drop, or when dropping would remove an image-saving node (then keep it as is)."""
    drop = set(unload_nodes(wf))
    if not drop:
        return None
    changed = True
    while changed:                                  # consumers of a dropped node lose their input -> drop them too
        changed = False
        for nid, n in wf.items():
            if nid in drop:
                continue
            if any(isinstance(v, list) and len(v) == 2 and str(v[0]) in drop for v in (n.get("inputs") or {}).values()):
                drop.add(nid)
                changed = True
    if any("save" in wf[nid].get("class_type", "").lower() for nid in drop):
        return None
    return {nid: n for nid, n in wf.items() if nid not in drop}


# ----------------------------------------------------------------------------------------------- presets
PRESET_DEFAULTS = {"backend": "comfyui", "workflow": "", "batch": 1, "seed_mode": "random", "seed": 0,
                   "width": 0, "height": 0, "negative": ""}


def list_presets() -> List[Dict[str, Any]]:
    os.makedirs(PRESET_DIR, exist_ok=True)
    out = []
    for n in sorted(os.listdir(PRESET_DIR)):
        if n.endswith(".json"):
            try:
                with open(os.path.join(PRESET_DIR, n), "r", encoding="utf-8") as h:
                    out.append({**PRESET_DEFAULTS, **json.load(h), "name": n[:-5]})
            except (OSError, ValueError):
                continue
    return out


def save_preset(p: Dict[str, Any]) -> Dict[str, Any]:
    name = safe_name(p.get("name", ""))
    data = {k: p.get(k, v) for k, v in PRESET_DEFAULTS.items()}
    data["batch"] = max(1, min(8, int(data["batch"] or 1)))
    for k in ("seed", "width", "height"):
        data[k] = int(data[k] or 0)
    if not data["workflow"]:
        raise ValueError("choose a workflow")
    os.makedirs(PRESET_DIR, exist_ok=True)
    with open(os.path.join(PRESET_DIR, name + ".json"), "w", encoding="utf-8") as h:
        json.dump(data, h, indent=1)
    return {**data, "name": name}


def get_preset(name: str) -> Dict[str, Any]:
    for p in list_presets():
        if p["name"] == name:
            return p
    raise ValueError(f"render preset not found: {name}")


# ----------------------------------------------------------------------------------------------- renders index
def renders_for(keys: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    want, out = set(keys), {}
    if not os.path.isfile(INDEX):
        return out
    with open(INDEX, "r", encoding="utf-8") as h:
        for line in h:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("key") in want and os.path.isfile(r.get("path", "")):
                out.setdefault(r["key"], []).append(r)
    return out


# ----------------------------------------------------------------------------------------------- one-GPU swap
_SWAP_LOCK = threading.Lock()
SWAPPED: Dict[str, Any] = {}


def swap_out_llm() -> Optional[str]:
    """Unload LM Studio's model(s) and remember the first so the next prompt job can reload it."""
    from .jobs import make_client
    s = config.load_settings()
    client = make_client(s)
    loaded = client.loaded_models()
    if not loaded:
        return None
    m = loaded[0]
    with _SWAP_LOCK:
        SWAPPED.update({"model": m.id, "context": getattr(m, "loaded_context", 0) or s.get("load_context_length")})
    client.unload_all()
    return m.id


def reload_swapped(client) -> bool:
    """Called when a prompt job finds nothing loaded: bring back the model the renderer unloaded."""
    with _SWAP_LOCK:
        info = dict(SWAPPED)
        SWAPPED.clear()
    if not info.get("model"):
        return False
    client.load(info["model"], info.get("context") or None)
    return True


# ----------------------------------------------------------------------------------------------- job
class RenderJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {"running": False, "results": []}
        self.cancel = threading.Event()

    def status(self, since: int = 0) -> Dict[str, Any]:
        with self.lock:
            st = {k: v for k, v in self.state.items() if k != "results"}
            st["results"] = self.state.get("results", [])[since:]
            st["count"] = len(self.state.get("results", []))
            return st

    def stop(self) -> None:
        self.cancel.set()

    def start(self, items: List[Dict[str, Any]], preset_name: str) -> Dict[str, Any]:
        items = [i for i in items if (i.get("text") or "").strip()]
        if not items:
            raise ValueError("nothing to render")
        preset = get_preset(preset_name)
        mapping = load_mapping(preset["workflow"])
        if not mapping.get("prompt"):
            raise ValueError(f"workflow '{preset['workflow']}' has no prompt field mapped - open Settings > Workflows")
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("a render is already running")
            self.state = {"running": True, "done": 0, "total": len(items) * preset["batch"], "errors": 0,
                          "last_error": "", "results": [], "started": time.time(), "phase": "starting",
                          "current": ""}
            self.cancel = threading.Event()
        threading.Thread(target=self._run, args=(items, preset, mapping), daemon=True).start()
        return self.status()

    def _set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def _run(self, items, preset, mapping) -> None:
        s = config.load_settings()
        c = Comfy(s.get("comfyui_url"))
        try:
            base = load_workflow(preset["workflow"])
            lean = without_unload(base)                   # batch renders keep models loaded between images
            jobs_total = len(items) * preset["batch"]
            job_no = 0
            if s.get("render_gpu", "shared") == "shared":
                self._set(phase="unloading the LM Studio model (one GPU)")
                gone = swap_out_llm()
                if gone:
                    self._set(swapped=gone)
            day = _dt.date.today().isoformat()
            out_dir = os.path.join(RENDER_DIR, day)
            os.makedirs(out_dir, exist_ok=True)
            done = 0
            for it in items:
                text = " ".join(it["text"].split())
                key = prompt_key(text)
                for b in range(preset["batch"]):
                    if self.cancel.is_set():
                        raise InterruptedError
                    seed = (int(preset["seed"]) + b) if preset["seed_mode"] == "fixed" else random.randrange(1, 2 ** 48)
                    job_no += 1
                    last = job_no == jobs_total
                    wf = copy.deepcopy(base if (last or lean is None) else lean)   # unload only after the last image

                    def setf(role, value):
                        ref = mapping.get(role)
                        if ref and ref[0] in wf:
                            wf[ref[0]]["inputs"][ref[1]] = value
                    setf("prompt", text)
                    setf("seed", seed)
                    if preset.get("negative"):
                        setf("negative", preset["negative"])
                    w, h = int(preset.get("width") or 0), int(preset.get("height") or 0)
                    if it.get("canvas"):                      # layout prompts carry their own size "WxH"
                        try:
                            w, h = [int(x) for x in str(it["canvas"]).lower().split("x")]
                        except ValueError:
                            pass
                    if w and h:
                        setf("width", w)
                        setf("height", h)
                    self._set(phase="rendering", current=text[:80])
                    try:
                        pid = c.queue(wf)
                        entry = c.wait(pid, self.cancel)
                        images = [im for out in (entry.get("outputs") or {}).values() for im in out.get("images", [])
                                  if im.get("type") == "output"]
                        if not images:                       # only previews: take the last temp image
                            images = [im for out in (entry.get("outputs") or {}).values()
                                      for im in out.get("images", [])][-1:]
                        for k, im in enumerate(images):
                            data = c.image(im["filename"], im.get("subfolder", ""), im.get("type", "output"))
                            ext = os.path.splitext(im["filename"])[1] or ".png"
                            path = os.path.join(out_dir, f"{key}_{seed}_{k}{ext}")
                            with open(path, "wb") as fh:
                                fh.write(data)
                            rec = {"key": key, "prompt": text, "path": path, "seed": seed, "preset": preset["name"],
                                   "workflow": preset["workflow"], "comfy_file": im["filename"],
                                   "made": _dt.datetime.now().isoformat(timespec="seconds"), "card": it.get("card", "")}
                            with open(INDEX, "a", encoding="utf-8") as fh:
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            with self.lock:
                                self.state["results"].append(rec)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        with self.lock:
                            self.state["errors"] += 1
                            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                    done += 1
                    self._set(done=done)
            if s.get("render_free_after", True) and s.get("render_gpu", "shared") == "shared":
                self._set(phase="freeing ComfyUI VRAM")
                try:
                    c.free()
                except RuntimeError:
                    pass
        except InterruptedError:
            try:                                          # stopped mid-batch: the last (unloading) run never came
                c.free()
            except RuntimeError:
                pass
        except Exception as exc:
            traceback.print_exc()
            self._set(last_error=f"{type(exc).__name__}: {exc}")
        finally:
            self._set(running=False, phase="done", current="", finished=time.time())


JOB = RenderJob()
