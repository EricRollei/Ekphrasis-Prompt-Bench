"""
Shared job logic for the CLI and the GUI server.

  run_generation(settings, spec, on_event, cancel)  images -> prompts (cache, log, optional live per-image writes)
  write_outputs(settings, entries, ...)             finished prompts -> per-image / per-folder / combined files
  plan_outputs(settings, items, ...)                what the per-folder / combined files would be called

Events passed to on_event (dicts): start, load, item, done. The GUI streams them; the CLI prints them.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import config
from .generate import GenParams, PromptResult, generate_one
from .instructions import compose, load_modifiers, load_presets, parse_modifier_args
from .lmstudio import LMStudioClient, LMStudioError
from .outputs import (combined_name, group_file_names, per_image_exists, plan_groups, write_list,
                      write_per_image)
from .store import ResultCache, RunLog, append_index, save_instructions
from .model_caps import caps_for, modes_for, normalise_mode

OUT_MODES = ("per-image", "per-folder", "combined")


USE_LOADED = "__loaded__"      # model choice meaning "whatever LM Studio has loaded; never load or unload"


def make_client(settings: Dict) -> LMStudioClient:
    return LMStudioClient(settings["lmstudio_host"], settings["lms_exe"], settings["request_timeout_s"],
                          settings.get("lmstudio_models_dir", ""))


def resolve_model(client: LMStudioClient, model: str) -> Tuple[str, str, bool]:
    """
    -> (api_id, gguf_path, use_loaded). USE_LOADED picks the loaded vision model (first loaded model otherwise)
    and never loads or unloads anything.
    """
    if model == USE_LOADED:
        loaded = client.loaded_models()
        if not loaded:
            raise LMStudioError("Nothing is loaded in LM Studio - load a model there, or pick one in the list.")
        pick = next((m for m in loaded if m.vision), loaded[0])
        return pick.identifier or pick.id, pick.path, True
    path = next((m.path for m in client.list_models(include_all=True) if m.id == model), "")
    return model, path, False


def model_modes(client: LMStudioClient, model: str) -> Dict:
    """Reasoning modes available for a model (or the loaded one)."""
    try:
        api_id, path, _ = resolve_model(client, model)
    except LMStudioError:
        api_id, path = "", ""
    caps = caps_for(path)
    note = ("This model's chat template has a broken {REASON:} tag (candidate_key undefined), so its i-modes are "
            "unavailable; thinking is controlled with reasoning_effort instead (off = no thinking). Fix: paste the "
            "corrected template into LM Studio's prompt template setting for this model."
            if caps.get("tag_broken") else "")
    return {"model": api_id, "caps": caps, "note": note,
            "modes": [{"value": v, "label": l} for v, l in modes_for(caps)]}


@dataclass
class JobSpec:
    items: List[Tuple[str, str]]                       # (image, root)
    model: str
    preset: str = ""
    mods: List[str] = field(default_factory=list)      # "id" or "id=value"
    note: str = ""                                     # run-wide note
    notes: Dict[str, str] = field(default_factory=dict)  # per-image notes {image: note}
    variations: int = 1
    reasoning: str = ""
    max_tokens: int = 0
    temperature: Optional[float] = None
    seed: Optional[int] = None
    force: bool = False
    force_images: List[str] = field(default_factory=list)   # redo these even if cached (per-image "redo")
    skip_existing: bool = False
    load: bool = True
    live_per_image: bool = False                        # CLI: write .prompt.txt as each image finishes
    context_length: int = 0                             # context to load with (0 = settings load_context_length)
    facts: bool = False                                 # add measured facts (pose + face detectors) per image
    facts_ethnicity: bool = False
    layout: Dict[str, Any] = field(default_factory=dict)   # canvas for layout presets                       # include the detector's ethnicity guess in those facts
    argv: List[str] = field(default_factory=list)


FACTS_PREFIX = ("Measured by a head-pose detector - more reliable than your impression for which way heads are "
                "turned (left / right = side of the picture). State these directions in the prompt; describe "
                "everything else from the image: ")


def _with_facts(note: str, image: str, ethnicity: bool) -> str:
    try:
        from .facts import cached_analyze, direction_text
        text = direction_text(cached_analyze(image))               # head direction only (Nov 13 verdicts)
    except Exception as exc:                       # detectors optional - never block prompt writing
        print(f"WARNING: facts for {image}: {type(exc).__name__}: {exc}")
        return note
    if not text:
        return note
    return (note + "\n" if note else "") + FACTS_PREFIX + text


def run_generation(settings: Dict, spec: JobSpec, on_event: Callable[[Dict], None] = lambda e: None,
                   cancel: Optional[threading.Event] = None) -> Dict:
    presets = load_presets(config.PRESETS_DIR)
    preset_id = spec.preset or settings["default_preset"]
    if preset_id not in presets:
        raise KeyError(f"unknown preset {preset_id!r}; known: {', '.join(presets)}")
    preset = presets[preset_id]
    mods = parse_modifier_args(spec.mods, load_modifiers(config.MODIFIERS_DIR))
    base = compose(preset, mods, spec.note, spec.variations)

    client = make_client(settings)
    api_id, gguf_path, use_loaded = resolve_model(client, spec.model)
    load_s = 0.0
    if spec.load and not use_loaded:
        on_event({"type": "load", "model": api_id, "state": "loading"})
        load_s = client.ensure_loaded(api_id, spec.context_length or settings["load_context_length"],
                                      settings["exclusive_load"])
        on_event({"type": "load", "model": api_id, "state": "ready", "load_s": round(load_s, 1)})
    else:
        on_event({"type": "load", "model": api_id, "state": "ready", "load_s": 0})
    caps = caps_for(gguf_path)
    mode = normalise_mode(spec.reasoning or preset.reasoning, caps)
    spec.model = api_id                                   # events, logs and provenance record the real model

    params = GenParams(layout=dict(spec.layout or {}), model=api_id, max_tokens=spec.max_tokens, temperature=spec.temperature, seed=spec.seed,
                       reasoning=mode, max_side=settings["max_side"], snap=settings["snap"],
                       jpeg_quality=settings["jpeg_quality"], reasoning_budget=settings["reasoning_budget"], caps=caps)
    cache = ResultCache(config.CACHE_DIR)
    log = RunLog(config.RUNS_DIR, "run")
    save_instructions(config.RUNS_DIR, base.sha, {"system": base.system, "user": base.user, **base.parts})
    log.write({"event": "start", "argv": spec.argv, "model": spec.model, "reasoning_mode": mode, "caps": caps,
               "preset": preset.id, "mods": spec.mods,
               "note": spec.note, "instr_sha": base.sha, "system": base.system, "user": base.user,
               "load_s": round(load_s, 2), "images": len(spec.items)})

    suffix = settings["per_image_suffix"]
    results: Dict[str, PromptResult] = {}
    written: List[str] = []
    total = len(spec.items)
    on_event({"type": "start", "total": total, "log": log.path, "instr_sha": base.sha})
    t0 = time.perf_counter()
    forced = {os.path.normcase(p) for p in spec.force_images}
    for i, (image, root) in enumerate(spec.items, 1):
        if cancel is not None and cancel.is_set():
            on_event({"type": "cancelled", "i": i})
            break
        if spec.skip_existing and per_image_exists(image, suffix, settings["per_image_suffix_json"]):
            on_event({"type": "item", "i": i, "total": total, "image": image, "status": "skipped"})
            continue
        note = spec.notes.get(image, "").strip()
        if spec.facts:
            note = _with_facts(note, image, spec.facts_ethnicity)
        composed = compose(preset, mods, (spec.note + ("\n" if spec.note and note else "") + note).strip(),
                           spec.variations) if note else base
        if composed is not base:
            save_instructions(config.RUNS_DIR, composed.sha, {"system": composed.system, "user": composed.user,
                                                              **composed.parts})
        r = generate_one(client, image, preset, composed, params, cache, log,
                         force=spec.force or os.path.normcase(image) in forced)
        results[image] = r
        if spec.live_per_image and r.ok:
            path = write_per_image(image, r.prompts, suffix, settings["per_image_suffix_json"])
            written.append({"path": path, "action": "new", "count": len(r.prompts)})
            append_index(config.RUNS_DIR, [{"file": path, "mode": "per-image", "image": image, "prompt": p,
                                            "model": spec.model, "preset": preset.id, "instr": composed.sha}
                                           for p in r.prompts])
        on_event({"type": "item", "i": i, "total": total, "image": image,
                  "status": "error" if r.error else ("cached" if r.cached else "ok"),
                  "prompts": r.prompts, "stats": r.stats, "error": r.error,
                  "model": spec.model, "preset": preset.id, "instr_sha": composed.sha, "mode": mode,
                  "retried": r.retried, "over_limit": r.over_limit})
    elapsed = time.perf_counter() - t0
    errors = sum(1 for r in results.values() if r.error)
    summary = {"results": results, "written": written, "errors": errors, "elapsed_s": round(elapsed, 1),
               "preset": preset.id, "instr_sha": base.sha,
               "log": log.path, "log_obj": log, "load_s": round(load_s, 1)}
    on_event({"type": "done", "errors": errors, "elapsed_s": round(elapsed, 1), "log": log.path,
              "done": sum(1 for r in results.values() if r.ok)})
    return summary


def plan_outputs(settings: Dict, items: Sequence[Tuple[str, str]], group_by: str = "folder",
                 overrides: Optional[Dict[str, str]] = None) -> Dict:
    s = dict(settings)
    if overrides:
        s["group_name_overrides"] = {**settings.get("group_name_overrides", {}), **overrides}
    groups = plan_groups(items, s, group_by)
    names = group_file_names(groups, s)
    return {"groups": [{"root": root, "group": g, "name": names[(root, g)], "count": len(imgs)}
                       for (root, g), imgs in groups.items()],
            "combined": combined_name(items, s), "_groups": groups, "_names": names}


def write_outputs(settings: Dict, items: Sequence[Tuple[str, str]], prompts: Dict[str, List[str]], modes: List[str],
                  fmt: str = "lines", group_by: str = "folder", group_dir: str = "", combined: str = "",
                  overrides: Optional[Dict[str, str]] = None, on_exists: str = "new",
                  dedupe: bool = False, meta: Optional[Dict[str, Dict]] = None, sidecar: bool = False) -> List[Dict]:
    """
    Write files for finished prompts. Returns [{path, action: new|appended|unchanged, count, sidecar}].
    combined: file name (placed in group_dir) or full path; blank = automatic name.
    on_exists applies to list files (per-folder + combined); per-image files are always new (never overwritten).
    meta: {image: {model, preset, instr, at}} - goes into .jsonl lists, sidecars and runs/saved_index.jsonl.
    """
    bad = [m for m in modes if m not in OUT_MODES]
    if bad:
        raise ValueError(f"unknown output mode(s): {bad}")
    meta = meta or {}
    group_dir = group_dir or settings["group_dir"]
    ext = ".jsonl" if fmt == "jsonl" else ".txt"
    written: List[Dict] = []
    index: List[Dict] = []

    def note_index(path: str, mode: str, pairs) -> None:
        for image, p in pairs:
            m = meta.get(image) or {}
            index.append({"file": path, "mode": mode, "image": image, "prompt": p, "model": m.get("model", ""),
                          "preset": m.get("preset", ""), "instr": m.get("instr", ""), "made": m.get("at", "")})

    def add_list(result: Dict, mode: str) -> None:
        note_index(result["path"], mode, result.pop("_written"))
        written.append(result)

    done = [(img, root) for img, root in items if prompts.get(img)]
    if "per-image" in modes:
        for img, _ in done:
            path = write_per_image(img, prompts[img], settings["per_image_suffix"], settings["per_image_suffix_json"])
            written.append({"path": path, "action": "new", "count": len(prompts[img]), "sidecar": None})
            note_index(path, "per-image", [(img, p) for p in prompts[img]])
    if "per-folder" in modes and done:
        plan = plan_outputs(settings, done, group_by, overrides)
        for (root, g), imgs in plan["_groups"].items():
            name = plan["_names"][(root, g)]
            if ext != ".txt":
                name = name[:-4] + ext
            add_list(write_list(os.path.join(group_dir, name), [(img, prompts[img]) for img in imgs],
                                fmt, root, on_exists, dedupe, meta, sidecar), "per-folder")
    if "combined" in modes and done:
        roots = list(dict.fromkeys(r for _, r in done))
        try:
            base = os.path.commonpath(roots) if len(roots) > 1 else roots[0]
        except ValueError:                                 # roots on different drives
            base = roots[0]
        name = (combined or "").strip() or combined_name(done, settings, ext)
        if not os.path.splitext(name)[1]:
            name += ext
        path = name if os.path.isabs(name) else os.path.join(group_dir, name)
        add_list(write_list(path, [(img, prompts[img]) for img, _ in done], fmt, base, on_exists, dedupe, meta, sidecar),
                 "combined")
    append_index(config.RUNS_DIR, index)
    return written
