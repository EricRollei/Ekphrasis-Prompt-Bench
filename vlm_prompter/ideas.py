"""
Ideas: prompt lists WITHOUT images, for any target preset.

Modes
  brief      the model writes N distinct takes on a brief ("lantern procession on a frozen lake"); each take varies
             subject, setting, era, style - no two alike.
  dice       per idea the app rolls random facets from built-in pools (genre, era, medium, light, palette, mood,
             camera, place) - optionally on top of a brief - and the model builds the picture from the roll.
             Creative accidents without images (same idea as photobash's draw mode).
  wildcards  a template with __name__ (.txt files in the wildcards folder, setting wildcards_dir) and {a|b|c}
             alternatives is filled at random; "polish" lets the model turn each filled template into a full prompt
             in the preset's format, otherwise the filled text is the prompt.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from . import config

DICE = {
    "genre": ["fashion editorial", "film noir still", "fairy tale illustration", "science fiction", "documentary photograph",
              "gothic romance", "western", "cyberpunk", "folk tale", "surrealist painting", "sports photograph",
              "religious icon", "children's book", "horror", "mythology", "travel photograph", "music album cover",
              "botanical study", "street photography", "opera stage scene", "solarpunk", "art nouveau poster"],
    "era": ["ancient Egypt", "classical Greece", "Byzantine", "medieval", "Renaissance", "Baroque", "Edo-period Japan",
            "Victorian", "Belle Epoque", "1920s", "1940s wartime", "1950s", "1960s", "1970s", "1980s", "1990s",
            "early 2000s", "present day", "near future", "far future", "timeless"],
    "medium": ["35mm film photograph", "medium-format photograph", "Polaroid", "oil painting", "watercolour",
               "gouache illustration", "ink drawing", "woodblock print", "charcoal drawing", "fresco", "3D render",
               "claymation still", "anime cel art", "stained glass", "mosaic", "collagraph print", "pastel drawing",
               "tempera on panel", "cyanotype", "infrared photograph"],
    "light": ["golden hour backlight", "overcast soft light", "hard noon sun", "candlelight", "neon at night",
              "moonlight", "window light from one side", "stage spotlight", "fog-diffused dawn", "lightning flash",
              "firelight", "fluorescent office light", "rim light against darkness", "dappled forest light"],
    "palette": ["muted earth tones", "complementary teal and orange", "monochrome blue", "black and white",
                "pastel candy colours", "deep jewel tones", "sepia", "acid neon", "faded film colours", "red and black",
                "ivory and gold", "forest greens", "desaturated with one red accent"],
    "mood": ["serene", "melancholy", "triumphant", "eerie", "playful", "tender", "tense", "dreamlike", "defiant",
             "lonely", "ecstatic", "solemn", "mischievous", "nostalgic"],
    "camera": ["extreme close-up", "close-up portrait", "medium shot", "full-length shot", "wide establishing shot",
               "low angle looking up", "high angle looking down", "overhead view", "Dutch angle", "over-the-shoulder",
               "symmetrical frontal view", "telephoto compression", "fisheye"],
    "place": ["a frozen lake", "a rooftop at night", "a desert canyon", "a flooded cathedral", "a night market",
              "a greenhouse", "a train carriage", "a lighthouse", "a salt flat", "a library", "a kelp forest",
              "an abandoned theatre", "a mountain monastery", "a laundromat", "a carnival", "a glacier cave",
              "a tea house", "a junkyard", "a ballroom", "a rice terrace"],
}

IDEA_SYSTEM = """You are an imaginative art director inventing pictures. There is NO image: you invent each picture
from the brief and directions you are given, then write its prompt.

Make each picture specific and visual: a concrete subject doing something, in a concrete place, with a definite
light, palette, medium and camera. Surprising but believable - one coherent scene, never a collage. Avoid cliches
unless the brief asks for them.

{mode_rules}

The target model and its prompt format are described next. Follow its format, order and length rules, but ignore
any instruction there about looking at or recreating an image - you are describing a picture you invented."""

BRIEF_RULES = """This is take {n} of {total} on the same brief. Make it clearly different from typical takes: choose
your own angle on subject, setting, era, style and mood. Earlier takes in this batch: {earlier}"""

DICE_RULES = """Build the picture from these rolled ingredients (use all of them; make them belong together - the
odd combinations are the point):
{roll}"""

POLISH_RULES = """Turn this rough idea into a full picture and write its prompt.

ROUGH IDEA (the user's template with its wildcards already filled in):
{seed}

{picks}

Everything in the rough idea that is not listed above as a randomly selected element is the user's own wording:
keep its meaning and intent exactly (you may improve the phrasing). Add concrete, visual detail that fits all of it -
one coherent picture."""


# ----------------------------------------------------------------------------------------------- wildcards
class Wildcards:
    """Name index over a wildcards folder: __name__ matches a file stem; __sub/dir/name__ matches a relative path."""

    def __init__(self, root: str):
        self.root = root
        self.by_stem: Dict[str, str] = {}
        self.by_rel: Dict[str, str] = {}
        self._cache: Dict[str, List[str]] = {}
        for base, dirs, names in os.walk(root):
            if "_prompts" in base:
                continue
            for n in names:
                if n.lower().endswith(".txt"):
                    p = os.path.join(base, n)
                    rel = os.path.relpath(p, root)[:-4].replace("\\", "/").lower()
                    self.by_rel[rel] = p
                    self.by_stem.setdefault(n[:-4].lower(), p)

    def lines(self, name: str) -> List[str]:
        key = name.strip().lower().replace("\\", "/")
        p = self.by_rel.get(key) or self.by_stem.get(key) or self.by_stem.get(key.split("/")[-1])
        if not p:
            raise KeyError(f"wildcard not found: __{name}__")
        if p not in self._cache:
            with open(p, "r", encoding="utf-8", errors="replace") as h:
                self._cache[p] = [l.strip() for l in h if l.strip() and not l.lstrip().startswith("#")]
        return self._cache[p]

    def search(self, text: str, limit: int = 40) -> List[str]:
        t = text.strip().lower()
        hits = [r for r in self.by_rel if t in r] if t else []
        return sorted(hits, key=lambda r: (len(r), r))[:limit]

    def fill(self, template: str, rng: random.Random, depth: int = 0) -> str:
        if depth > 8:
            return template
        out = re.sub(r"\{([^{}]*\|[^{}]*)\}", lambda m: rng.choice(m.group(1).split("|")).strip(), template)
        out = re.sub(r"__([^_\s][^\s]*?)__", lambda m: rng.choice(self.lines(m.group(1)) or [""]), out)
        return self.fill(out, rng, depth + 1) if ("__" in out or "|" in out) and out != template else out


_WC: Dict[str, Wildcards] = {}
_WC_LOCK = threading.Lock()


def wildcards(root: str) -> Wildcards:
    if not root or not os.path.isdir(root):
        raise ValueError(f"wildcards folder not found: {root or '(not set)'} - set wildcards_dir in settings.json "
                         "to a folder of .txt wildcard files")
    with _WC_LOCK:
        if root not in _WC:
            _WC[root] = Wildcards(root)
        return _WC[root]


def roll(rng: random.Random, facets: List[str]) -> Dict[str, str]:
    return {f: rng.choice(DICE[f]) for f in facets if f in DICE}


# ----------------------------------------------------------------------------------------------- job
class IdeasJob:
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
        mode = spec.get("mode")
        if mode not in ("brief", "dice", "wildcards"):
            raise ValueError("mode must be brief, dice or wildcards")
        if mode == "brief" and not (spec.get("brief") or "").strip():
            raise ValueError("write a brief first")
        if mode == "wildcards" and not (spec.get("template") or "").strip():
            raise ValueError("write a template first")
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("an ideas run is already going")
            self.state = {"running": True, "done": 0, "total": int(spec.get("count") or 8), "errors": 0,
                          "last_error": "", "results": [], "started": time.time()}
            self.cancel = threading.Event()
        threading.Thread(target=self._run, args=(settings, spec), daemon=True).start()
        return self.status()

    def _run(self, settings, spec) -> None:
        from .generate import extract_json
        from .instructions import compose, load_presets
        try:
            mode = spec["mode"]
            seed = str(spec.get("seed") or "").strip()
            rng = random.Random(int(seed) if seed.lstrip("-").isdigit() else None)
            presets = load_presets(config.PRESETS_DIR)
            preset = presets[spec.get("preset") or settings["default_preset"]]
            composed = compose(preset, [], "", 1)
            use_model = True                      # polish is automatic (Nov 15): wildcards are always written up
            client = api_id = caps = mode_r = None
            if use_model:
                from .jobs import make_client, resolve_model
                from .model_caps import caps_for, normalise_mode
                client = make_client(settings)
                api_id, gguf, _ = resolve_model(client, spec.get("model") or "__loaded__")
                caps = caps_for(gguf)
                mode_r = normalise_mode(spec.get("reasoning") or preset.reasoning, caps)
                with self.lock:
                    self.state["model"] = api_id
            from . import wc as wcmod
            providers = {"llm": wcmod.make_llm_provider(client, api_id, caps)} if mode == "wildcards" else {}
            if mode == "wildcards":
                wcmod.engine(settings)                # fail early when the wildcards folder is not set
            brief = (spec.get("brief") or "").strip()
            facets = [f for f in (spec.get("dice") or list(DICE)) if f in DICE]
            temp = float(spec.get("temperature") or 1.0)
            earlier: List[str] = []
            total = self.state["total"]
            for n in range(1, total + 1):
                if self.cancel.is_set():
                    break
                try:
                    info: Dict[str, Any] = {}
                    if mode == "wildcards":
                        res = wcmod.process(settings, spec["template"], seed=rng.randrange(0, 0x7FFFFFFF),
                                            providers=providers)
                        if res.blocked:
                            raise ValueError(f"content filter blanked this fill ({res.blocked})")
                        info["filled"] = res.text
                        info["picks"] = [f"{s.label or s.source}: {s.value}" for s in res.top_level_slots()]
                        rules = POLISH_RULES.format(seed=res.text, picks=res.format_for_llm() or
                                                    "(no random picks - all of it is the user's own words)")
                    elif mode == "dice":
                        r = roll(rng, facets)
                        info["roll"] = r
                        rules = DICE_RULES.format(roll="\n".join(f"- {k}: {v}" for k, v in r.items()))
                        if brief:
                            rules = "Brief: " + brief + "\n" + rules
                    else:
                        rules = "Brief: " + brief + "\n" + BRIEF_RULES.format(
                            n=n, total=total, earlier="; ".join(earlier[-6:]) or "none yet")
                    system = IDEA_SYSTEM.format(mode_rules=rules) + "\n\n" + composed.system
                    msgs = [{"role": "system", "content": system},
                            {"role": "user", "content": "Invent the picture and write its prompt now."}]
                    extra = 4096 if mode_r not in ("off", "") and not mode_r.startswith("i") else 0
                    r = client.chat(api_id, msgs, max_tokens=preset.max_tokens + extra, temperature=temp,
                                    top_p=preset.top_p, seed=rng.randrange(1 << 30), reasoning=mode_r, caps=caps)
                    text = (r.content or "").strip()
                    if preset.output == "json":
                        text = extract_json(text)
                        if preset.layout:
                            from .layout import apply as apply_layout
                            _img = None
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
                    earlier.append(" ".join(text.split()[:12]) + " ...")
                    self._add(n, text, mode, preset.id, api_id, info)
                except Exception as exc:
                    with self.lock:
                        self.state["errors"] += 1
                        self.state["last_error"] = f"idea {n}: {type(exc).__name__}: {exc}"
                finally:
                    with self.lock:
                        self.state["done"] = n
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.state.update(running=False, finished=time.time())

    def _add(self, n, text, mode, preset_id, model, info) -> None:
        with self.lock:
            from .prompt_grades import grade_of
            info = {**info, "flags": ["graded duff before"] if grade_of(text) == "duff" else []}
            self.state["results"].append({"n": n, "prompt": text, "mode": mode, "preset": preset_id, "model": model,
                                          "images": [], "weights": [], "takes": [], "assignment": {}, **info})


def save_results(settings: Dict[str, Any], results: List[Dict[str, Any]], name: str = "") -> str:
    """Same file layout as photobash lists; variant 'ideas <mode>'."""
    import datetime as _dt
    import hashlib
    import json
    from .outputs import unique_path
    from .store import append_index
    if not results:
        raise ValueError("nothing to save")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    name = (name or "").strip() or f"ideas_{stamp}"
    if not name.lower().endswith(".txt"):
        name += ".txt"
    path = unique_path(name if os.path.isabs(name) else os.path.join(settings["group_dir"], name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    made = _dt.datetime.now().isoformat(timespec="seconds")
    lines, meta, index = [], [], []
    for r in results:
        text = " ".join(r["prompt"].split())
        lines.append(text)
        rec = {"line": len(lines), "variant": f"ideas {r['mode']}", "preset": r["preset"], "model": r["model"],
               "roll": r.get("roll"), "filled": r.get("filled"),
               "prompt_sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]}
        meta.append(rec)
        index.append({"file": path, "mode": "ideas", "image": "", "prompt": text, "model": r["model"],
                      "preset": r["preset"], "instr": "", "variant": f"ideas {r['mode']}", "made": made})
    with open(path, "w", encoding="utf-8", newline="\n") as h:
        h.write("\n".join(lines) + "\n")
    with open(path[:-4] + ".meta.jsonl", "w", encoding="utf-8", newline="\n") as h:
        for rec in meta:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    append_index(config.RUNS_DIR, index)
    return path


JOB = IdeasJob()
