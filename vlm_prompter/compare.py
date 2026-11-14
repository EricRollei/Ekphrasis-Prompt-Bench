"""
Compare runs: several prompt VERSIONS per source image in one go, written to one prompt list for a sweep.

A version = preset + measured facts on/off + thinking mode (the loaded model, or one chosen in the header).
Every source image gets one prompt per version; the list keeps each image's versions next to each other
(sweep cells for one source sit side by side in Review). Every prompt goes into runs/saved_index.jsonl with its
"variant" label, so Review shows which version made a render and phase-3 reports can compare versions.

Files (never overwritten - a new numbered name when one exists):
  <group_dir>/<name>.txt          one prompt per line, in image order, versions together
  <group_dir>/<name>.meta.jsonl   one record per line: image, variant, model, preset, instr, prompt_sha

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .outputs import unique_path
from .store import append_index


def variant_label(v: Dict[str, Any]) -> str:
    parts = [v.get("preset", "")]
    if v.get("facts"):
        parts.append("facts")
    if v.get("reasoning") and v["reasoning"] not in ("off", ""):
        parts.append(v["reasoning"])
    return " + ".join(p for p in parts if p)


def one_line(text: str) -> str:
    return " ".join((text or "").split())


class CompareJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {"running": False}
        self.cancel = threading.Event()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def stop(self) -> None:
        self.cancel.set()

    def start(self, settings: Dict[str, Any], items: List[Tuple[str, str]], variants: List[Dict[str, Any]],
              base: Dict[str, Any], name: str = "") -> Dict[str, Any]:
        if not items or not variants:
            raise ValueError("need at least one image and one version")
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("a compare run is already going")
            self.state = {"running": True, "variant": 0, "variants": len(variants), "done": 0, "total": len(items),
                          "errors": 0, "last_error": "", "started": time.time(), "file": ""}
            self.cancel = threading.Event()
        threading.Thread(target=self._run, args=(settings, items, variants, base, name), daemon=True).start()
        return self.status()

    def _run(self, settings, items, variants, base, name) -> None:
        from .jobs import JobSpec, run_generation
        prompts: Dict[Tuple[str, str], Dict[str, Any]] = {}       # (image, label) -> {prompt, model, preset, instr}
        labels = [variant_label(v) for v in variants]
        try:
            for vi, (v, label) in enumerate(zip(variants, labels), 1):
                if self.cancel.is_set():
                    break
                with self.lock:
                    self.state.update(variant=vi, label=label, done=0)
                spec = JobSpec(items=list(items), model=base.get("model") or "__loaded__", preset=v.get("preset", ""),
                               mods=list(base.get("mods") or []), note=base.get("note", ""),
                               reasoning=v.get("reasoning", ""), max_tokens=int(base.get("max_tokens") or 0),
                               temperature=base.get("temperature"), seed=base.get("seed"),
                               facts=bool(v.get("facts")), layout=dict(base.get("layout") or {}), trigger=base.get("trigger", ""),
                               argv=["compare", label])

                def on_event(e, label=label):
                    if e.get("type") == "item":
                        with self.lock:
                            self.state["done"] = e["i"]
                            if e.get("status") == "error":
                                self.state["errors"] += 1
                                self.state["last_error"] = f"{os.path.basename(e['image'])}: {e.get('error')}"
                        if e.get("prompts"):
                            prompts[(e["image"], label)] = {"prompt": e["prompts"][0], "model": e.get("model", ""),
                                                            "preset": e.get("preset", ""), "instr": e.get("instr_sha", "")}

                run_generation(settings, spec, on_event, self.cancel)
            path = self._write(settings, items, labels, prompts, name)
            with self.lock:
                self.state["file"] = path
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.state.update(running=False, finished=time.time())

    @staticmethod
    def _write(settings, items, labels, prompts, name) -> str:
        if not prompts:
            return ""
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
        name = (name or "").strip() or f"compare_{stamp}"
        if not name.lower().endswith(".txt"):
            name += ".txt"
        path = unique_path(name if os.path.isabs(name) else os.path.join(settings["group_dir"], name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        made = _dt.datetime.now().isoformat(timespec="seconds")
        lines, meta, index = [], [], []
        for image, _root in items:                        # versions of one image next to each other
            for label in labels:
                r = prompts.get((image, label))
                if not r:
                    continue
                text = one_line(r["prompt"])
                lines.append(text)
                rec = {"line": len(lines), "image": image, "variant": label, "model": r["model"], "preset": r["preset"],
                       "instr": r["instr"], "prompt_sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]}
                meta.append(rec)
                index.append({"file": path, "mode": "compare", "image": image, "prompt": text, "model": r["model"],
                              "preset": r["preset"], "instr": r["instr"], "variant": label, "made": made})
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
        with open(path[:-4] + ".meta.jsonl", "w", encoding="utf-8", newline="\n") as handle:
            for rec in meta:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        append_index(config.RUNS_DIR, index)
        return path


JOB = CompareJob()
