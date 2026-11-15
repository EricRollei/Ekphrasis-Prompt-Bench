"""
Photobash: several pictures -> one NEW picture, written as a text prompt for any target preset.

Not a collage, not a double exposure: one coherent scene that fuses the sources. The language model has to invent
the bridge between pictures that do not belong together - that is where the creative accidents come from (Eric,
Nov 13: as DiT models and encoders improve, fewer happen by themselves).

Sources (2-4): an image, or a FOLDER = a pool (each mix draws a random image from it). Each source has
  weight  0-100   how much of the final picture it supplies (relative to the other sources)
  take    "everything", or a list of facets it contributes ONLY (style from one, setting from another ...)
Modes:
  blend   the model fuses all sources at once; weights say how strongly each shapes the result
  draw    per mix, every facet is drawn from one source with probability = weight (only sources that offer
          that facet); the model then builds the picture from that assignment. Many mixes = many combinations.
Output prompts follow the chosen preset's format and length rules (krea2, moss, ideogram4 JSON ...).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from . import config

FACETS = ["subject", "pose", "clothing", "setting", "light", "palette", "style", "genre", "era", "mood",
          "composition", "texture"]
FACET_WORDS = {
    "subject": "the main subject(s): who or what is shown",
    "pose": "pose, gesture, action and which way figures face",
    "clothing": "clothing, costume and accessories",
    "setting": "the setting, place and background",
    "light": "the lighting: sources, direction, quality, time of day",
    "palette": "the colour palette and tonal key",
    "style": "the medium and rendering style (photograph, oil paint, anime ...)",
    "genre": "the genre (fashion editorial, film still, fairy tale, sci-fi, documentary ...)",
    "era": "the time period / era (costume, objects, technology, photographic look of the period)",
    "mood": "the mood and emotional atmosphere",
    "composition": "the composition: framing, camera angle, shot size, where things sit in the frame",
    "texture": "surface textures and materials, grain, brushwork",
}

PB_SYSTEM = """You are a photobash artist. You are shown several numbered pictures and you invent ONE new picture that
fuses them, then write the prompt for it.

The new picture is a single coherent image - never a collage, never a double exposure, never pictures side by
side or pasted together. It has one scene, one perspective, one light logic, one medium. Everything in it must
belong together: the subject really stands in that setting, lit by that light, rendered in that style.

Where the pictures conflict, do not average them into something bland - resolve the conflict into a fresh idea
that keeps the strongest qualities of each (a knight's armour re-imagined in the fabric of a couture gown, a
desert lit like a neon street at night). Be bold and specific; surprising but believable beats safe.

{mode_rules}

In the prompt itself: describe only the new picture. Never mention "picture 1", the source pictures, weights,
mixing or the word photobash.

The target model and its prompt format are described next. Follow its format, order and length rules, but
ignore any instruction there about recreating ONE image - you are describing the new fused picture."""

BLEND_RULES = """How much each picture contributes is given as a weight. The heaviest picture is the backbone: its
scene structure and main subject dominate. Lighter pictures transform, colour, costume or re-stage it in
proportion to their weight. At equal weights make a true hybrid. A picture marked "take ONLY" contributes just
those aspects and nothing else."""

DRAW_RULES = """Each aspect of the new picture has been assigned to one source picture. Take that aspect from that
picture as faithfully as the other aspects allow, and invent whatever is needed to make them belong together.
Aspects not listed are yours to choose so the whole picture works."""


def _facets_of(src: Dict[str, Any]) -> List[str]:
    take = src.get("take") or "everything"
    return list(FACETS) if take == "everything" else [f for f in take if f in FACETS]


def draw_assignment(sources: List[Dict[str, Any]], rng: random.Random) -> Dict[str, int]:
    """facet -> source index, drawn with probability proportional to weight among sources offering the facet."""
    out: Dict[str, int] = {}
    for f in FACETS:
        cands = [(i, max(0.0, float(s.get("weight", 50)))) for i, s in enumerate(sources) if f in _facets_of(s)]
        cands = [(i, w) for i, w in cands if w > 0]
        if not cands:
            continue
        total = sum(w for _, w in cands)
        r, acc = rng.random() * total, 0.0
        for i, w in cands:
            acc += w
            if r <= acc:
                out[f] = i
                break
    return out


def pick_images(sources: List[Dict[str, Any]], rng: random.Random) -> List[str]:
    """One image per source: the image itself, or a random image from a folder pool."""
    from .images import is_image
    out = []
    for s in sources:
        paths = s.get("paths") or [s["path"]]
        folders = [f for f in paths if os.path.isdir(f)]
        files = [f for f in paths if os.path.isfile(f)] if len(paths) > 1 or folders else []
        if folders or files:
            pools = s.get("_pools")
            if pools is None:                      # per folder, so a big folder does not drown a small one;
                pools = s["_pools"] = [sorted(os.path.join(f, n) for n in os.listdir(f) if is_image(n)) for f in folders]
                if files:                          # a hand-picked selection counts as one more pool
                    pools.append(sorted(files))
                pools[:] = [pl for pl in pools if pl]
            if not pools:
                raise ValueError(f"no images in {', '.join(folders)}")
            out.append(rng.choice(rng.choice(pools)))      # folder first (equal chance), then an image in it
        else:
            out.append(s["path"])
    return out


def directions_text(spec: Dict[str, Any]) -> str:
    """Master directions for every mix (Nov 15): no text, no signatures / watermarks, add a given text."""
    out = []
    add = (spec.get("add_text") or "").strip().strip('"')
    if spec.get("avoid_text"):
        out.append("The new picture contains no text, lettering, signs, captions or typography"
                   + (" other than the requested text below" if add else "") + " - describe none.")
    if spec.get("avoid_marks"):
        out.append("No signatures, artist marks, monograms, watermarks, logos or interface overlays anywhere in the picture.")
    if add:
        where = (spec.get("text_where") or "").strip()
        out.append(f'Include this text in the picture, exactly as written: "{add}"' + (f", {where}" if where else "")
                   + ". Put it in double quotes in the prompt and describe its lettering style, colour and placement "
                     "so it belongs naturally in the scene.")
    return " ".join(out)


def build_messages(preset_system: str, preset_user: str, mode: str, sources: List[Dict[str, Any]],
                   images: List[str], assignment: Optional[Dict[str, int]], note: str = "",
                   image_url=None) -> List[Dict[str, Any]]:
    if image_url is None:
        from .vision import _image_url as image_url
    system = PB_SYSTEM.format(mode_rules=DRAW_RULES if mode == "draw" else BLEND_RULES) + "\n\n" + preset_system
    total = sum(max(0.0, float(s.get("weight", 50))) for s in sources) or 1.0
    content: List[Dict[str, Any]] = []
    for i, (s, img) in enumerate(zip(sources, images), 1):
        label = f"Picture {i}"
        if mode == "blend":
            share = round(100 * max(0.0, float(s.get("weight", 50))) / total)
            take = s.get("take") or "everything"
            label += f" - weight {share}%" + ("" if take == "everything" else
                                            " - take ONLY: " + "; ".join(FACET_WORDS[f] for f in _facets_of(s)))
        content.append({"type": "text", "text": label + ":"})
        content.append({"type": "image_url", "image_url": {"url": image_url(img)}})
    if mode == "draw" and assignment:
        lines = [f"- {FACET_WORDS[f]}: from Picture {i + 1}" for f, i in assignment.items()]
        content.append({"type": "text", "text": "Aspect assignment for this new picture:\n" + "\n".join(lines)})
    if note:
        content.append({"type": "text", "text": "Extra direction from the artist: " + note})
    closing = preset_user.strip()
    if "image" in closing.lower() or "picture" in closing.lower():   # "...for this image" contradicts a NEW picture
        closing = ""
    content.append({"type": "text", "text": ("Now write the prompt for the new fused picture. " + closing).strip()})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


class PhotobashJob:
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

    def start(self, settings: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        sources = [dict(s) for s in spec.get("sources") or [] if (s.get("path") or "").strip()]
        if len(sources) < 2:
            raise ValueError("photobash needs at least two sources (images or folders)")
        for s in sources:
            s["path"] = s["path"].strip().strip('"')
            s["paths"] = [p.strip().strip('"') for p in (s.get("paths") or [s["path"]]) if p.strip()]
            for p in s["paths"]:
                if not os.path.exists(p):
                    raise ValueError(f"not found: {p}")
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("a photobash run is already going")
            self.state = {"running": True, "done": 0, "total": int(spec.get("mixes") or 4), "errors": 0,
                          "last_error": "", "results": [], "started": time.time()}
            self.cancel = threading.Event()
        threading.Thread(target=self._run, args=(settings, sources, spec), daemon=True).start()
        return self.status()

    def _run(self, settings, sources, spec) -> None:
        from .instructions import compose, load_presets
        from .jobs import make_client, resolve_model
        from .model_caps import caps_for, normalise_mode
        from .generate import extract_json
        try:
            presets = load_presets(config.PRESETS_DIR)
            preset = presets[spec.get("preset") or settings["default_preset"]]
            composed = compose(preset, [], "", 1)
            client = make_client(settings)
            api_id, gguf, _ = resolve_model(client, spec.get("model") or "__loaded__")
            caps = caps_for(gguf)
            mode_r = normalise_mode(spec.get("reasoning") or preset.reasoning, caps)
            mode = spec.get("mode") if spec.get("mode") in ("blend", "draw") else "blend"
            seed = spec.get("seed")
            rng = random.Random(int(seed) if str(seed or "").strip().lstrip("-").isdigit() else None)
            temp = float(spec.get("temperature") or 0.9)
            with self.lock:
                self.state["model"] = api_id
            for n in range(1, self.state["total"] + 1):
                if self.cancel.is_set():
                    break
                try:
                    fixed = spec.get("fixed") or {}
                    if fixed.get("images"):                 # redo of one mix: same pictures, same aspect assignment
                        images = list(fixed["images"])
                        fa = fixed.get("assignment") or {}
                        assignment = {f: int(i) - 1 for f, i in fa.items()} if (mode == "draw" and fa) else \
                            (draw_assignment(sources, rng) if mode == "draw" else None)
                    else:
                        images = pick_images(sources, rng)
                        assignment = draw_assignment(sources, rng) if mode == "draw" else None
                    note = " ".join(x for x in (directions_text(spec), spec.get("note", "")) if x)
                    msgs = build_messages(composed.system, composed.user, mode, sources, images, assignment, note)
                    extra = 4096 if mode_r not in ("off", "") and not mode_r.startswith("i") else 0
                    mt = int(spec.get("max_tokens") or 0) or preset.max_tokens
                    from .generate import runaway
                    flags = []
                    for attempt in range(2):                # one silent retry when the model falls into a loop
                        r = client.chat(api_id, msgs, max_tokens=mt + extra, temperature=temp,
                                        top_p=preset.top_p, seed=rng.randrange(1 << 30), reasoning=mode_r, caps=caps)
                        text = (r.content or "").strip()
                        why = runaway(text) if preset.output != "json" else ""
                        if not why:
                            break
                        flags = [why + (" - retried once, still bad: press redo" if attempt else "")]
                    if not why:
                        flags = []
                    if preset.output == "json":
                        text = extract_json(text)
                        if preset.layout:
                            from .layout import apply as apply_layout
                            _img = images[max(range(len(images)), key=lambda i: float(sources[i].get('weight', 50) or 0))]
                            _wh = None
                            if _img:
                                from PIL import Image as _I
                                with _I.open(_img) as _im:
                                    _wh = _im.size
                            text, _canvas = apply_layout(preset.layout, text, spec.get('layout'), _wh)
                            text = text + ''
                    if not text:
                        raise ValueError("empty answer")
                    if spec.get('trigger'):
                        from .instructions import apply_trigger
                        text = apply_trigger(text, spec['trigger'], preset.output)
                    res = {"n": n, "prompt": text, "mode": mode, "preset": preset.id, "model": api_id,
                           "images": images, "weights": [s.get("weight", 50) for s in sources],
                           "takes": [s.get("take") or "everything" for s in sources],
                           "assignment": {f: i + 1 for f, i in (assignment or {}).items()},
                           "tokens": getattr(r, "completion_tokens", None), "flags": flags,
                           "replaces": fixed.get("replaces")}
                    with self.lock:
                        self.state["results"].append(res)
                except Exception as exc:
                    with self.lock:
                        self.state["errors"] += 1
                        self.state["last_error"] = f"mix {n}: {type(exc).__name__}: {exc}"
                with self.lock:
                    self.state["done"] = n
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.state.update(running=False, finished=time.time())


def save_results(settings: Dict[str, Any], results: List[Dict[str, Any]], name: str = "") -> str:
    """One prompt per line + .meta.jsonl; provenance records (variant 'photobash <mode>', all source images)."""
    from .outputs import unique_path
    from .store import append_index
    if not results:
        raise ValueError("nothing to save")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    name = (name or "").strip() or f"photobash_{stamp}"
    if not name.lower().endswith(".txt"):
        name += ".txt"
    path = unique_path(name if os.path.isabs(name) else os.path.join(settings["group_dir"], name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    made = _dt.datetime.now().isoformat(timespec="seconds")
    lines, meta, index = [], [], []
    for r in results:
        text = " ".join(r["prompt"].split())
        lines.append(text)
        main = max(range(len(r["images"])), key=lambda i: float(r["weights"][i] or 0))
        rec = {"line": len(lines), "variant": f"photobash {r['mode']}", "preset": r["preset"], "model": r["model"],
               "images": r["images"], "weights": r["weights"], "takes": r["takes"], "assignment": r["assignment"],
               "prompt_sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]}
        meta.append(rec)
        index.append({"file": path, "mode": "photobash", "image": r["images"][main], "prompt": text, "model": r["model"],
                      "preset": r["preset"], "instr": "", "variant": f"photobash {r['mode']}", "sources": r["images"],
                      "made": made})
    with open(path, "w", encoding="utf-8", newline="\n") as h:
        h.write("\n".join(lines) + "\n")
    with open(path[:-4] + ".meta.jsonl", "w", encoding="utf-8", newline="\n") as h:
        for rec in meta:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    append_index(config.RUNS_DIR, index)
    return path


JOB = PhotobashJob()


def preview(settings: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Exactly what one mix sends to the model, with the pictures shown as placeholders (no model, no images)."""
    from .instructions import compose, load_presets
    preset = load_presets(config.PRESETS_DIR)[spec.get("preset") or settings["default_preset"]]
    composed = compose(preset, [], "", 1)
    sources = [dict(s) for s in spec.get("sources") or [] if (s.get("path") or "").strip()] or \
        [{"path": "picture-1", "weight": 50}, {"path": "picture-2", "weight": 50}]
    mode = spec.get("mode") if spec.get("mode") in ("blend", "draw") else "blend"
    assignment = draw_assignment(sources, random.Random(0)) if mode == "draw" else None
    note = " ".join(x for x in (directions_text(spec), spec.get("note", "")) if x)
    msgs = build_messages(composed.system, composed.user, mode, sources, [s["path"] for s in sources], assignment,
                          note, image_url=lambda p: "")
    user = []
    k = 0
    for part in msgs[1]["content"]:
        if part["type"] == "text":
            user.append(part["text"])
        else:
            k += 1
            user.append(f"[picture {k} goes here]")
    return {"system": msgs[0]["content"], "user": "\n".join(user), "preset": preset.id,
            "max_tokens": int(spec.get("max_tokens") or 0) or preset.max_tokens,
            "draw_example": mode == "draw"}
