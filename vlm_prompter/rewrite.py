"""
Rewrite: take existing prompts and rewrite them - always into the format and length of the chosen preset.

Operations
  convert     format / length only (SDXL tag prompts -> Krea 2 / MOSS prose, Krea 2 -> Ideogram JSON ...)
  edit        an instruction applied to every prompt ("change the style to a professional photograph",
              "more surreal", "happier", "replace the woman with an old fisherman")
  refine      the prompt is checked against its source image (needs the image: provenance or queue) and corrected
  rules       one or more lessons applied to every prompt ("state which way each person faces")
  variations  N rewrites of each prompt, each a different wording / take

Word choice (instruction-based, Eric's finding - no sampler changes): faithful | fresh | bold.
Review by exception: each result gets flags only when something looks wrong (protected words or quoted text lost,
over the token limit, a preamble, unchanged, far too short).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from . import config

RW_SYSTEM = """You are an expert prompt writer rewriting an existing text-to-image prompt.

The INPUT PROMPT may be prose, a tag list (SDXL / booru style, possibly with weights like (word:1.3) or [word],
score_9 / masterpiece quality tags, BREAK separators) or JSON. Read it for its content:
- keep every subject, object, attribute, action, place, style cue and relation it contains unless the operation
  below changes it;
- turn tags into clear natural phrases; turn weights into emphasis by wording, never by syntax; drop pure quality
  and score tags (masterpiece, best quality, score_9, 8k ...);
- where the target format wants more detail than the input has, add only concrete, visual detail that is
  consistent with everything already there - never contradict it, never pad with vague praise;
- keep the result clear: one coherent picture, each attribute attached to its own subject.

OPERATION:
{op_rules}

WORD CHOICE:
{word_rules}

{keep_rules}Output only the rewritten prompt - no preamble ("Here is ..."), no notes, no quotes around it.

The target model and its prompt format are described next. Follow its format, order and length rules exactly,
but ignore any instruction there about looking at an image unless an image is attached to this request."""

OPS = {
    "convert": "Rewrite the input prompt into the target format below, keeping its content.",
    "edit": """Apply this change to the picture, then write the full prompt for the changed picture:
  "{instruction}"
Carry the change through EVERY part it affects, not only the first sentence. A style or medium change rewrites all
style-bound detail consistently (brushwork, linework and paper texture become skin, fabric and lens detail for a
photograph, and the reverse; period palettes become real light, and the reverse). A subject change keeps the
pose, setting, light and composition unless the instruction says otherwise. A mood change ("happier", "more
surreal") shows in visible things - expression, gesture, light, colour, objects - not in mood words alone.
Everything the instruction does not touch stays as it was.""",
    "refine": """An image is attached: the picture this prompt is meant to describe. Compare the prompt with the image and
correct it: fix anything wrong (colours, counts, clothing, which way people face, left/right placement, what hands
touch), add important visible things that are missing, remove things that are not in the image. Keep everything
that is already right, including its wording where possible.{facts}""",
    "rules": """Apply these rules to the prompt while keeping its content:
{rules}""",
    "pair": """Make a TWIN of the input prompt for a blend / concept-direction pair. Apply only this change:
  "{instruction}"
Change ONLY the words that must change for it. Copy every other word, phrase, comma and the sentence order exactly
from the input, so the two prompts differ in as few words as possible. Do not improve, reword, reorder or expand
anything else, and do not apply the target format's length rules - the twin must match the input's length.""",
    "variations": """This is variation {k} of {n}: keep the same picture, but write it as a clearly different take -
different wording, different emphasis and sentence order, a different angle on the same scene. Earlier variations
began: {earlier}""",
}

WORDS = {
    "faithful": "Keep the input's own words wherever they still fit; change only what the operation and the target "
                "format require.",
    "fresh": "Avoid the most predictable phrasing. Prefer specific, less common but accurate words and fresh, concrete "
             "images over stock phrases (not 'stunning', 'breathtaking', 'ethereal glow', 'vibrant colors', "
             "'intricate details'). Every word must still describe something visible.",
    "bold": "Write as a distinctive art director with a voice: precise, unexpected vocabulary, striking concrete "
            "comparisons for textures, light and materials, no stock phrases at all. Stay accurate - bold wording, "
            "never invented content beyond what the operation allows.",
}

EDIT_CHIPS = ["change the style to a professional photograph", "change the style to a vintage illustration",
              "make it more surreal", "make it happier", "make it darker and more dramatic", "change the era to the 1920s",
              "replace the main subject with ..."]

_PREAMBLE = re.compile(r"^\s*(here(?:'s| is)|sure|certainly|okay|prompt\s*:|rewritten prompt)", re.I)


def split_text(text: str) -> List[str]:
    """Pasted text -> prompts: blank-line separated paragraphs if there are any, else one per line."""
    t = (text or "").replace("\r\n", "\n").strip()
    if not t:
        return []
    parts = re.split(r"\n\s*\n", t) if re.search(r"\n\s*\n", t) else t.split("\n")
    return [" ".join(p.split()) for p in parts if p.strip()]


def load_list(path: str) -> List[Dict[str, Any]]:
    """A prompt list file (.txt one per line, .jsonl with a prompt field) -> items with source images when known."""
    from .review import index_lookup
    path = path.strip().strip('"')
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as h:
        raw = h.read()
    if path.lower().endswith(".jsonl"):
        for line in raw.splitlines():
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("prompt"):
                items.append({"text": o["prompt"], "image": o.get("image", "")})
    else:
        items = [{"text": t, "image": ""} for t in split_text(raw)]
    look = index_lookup()
    for it in items:
        if not it["image"]:
            prov = look.find(it["text"]) or {}
            it["image"] = prov.get("source_image", "")
    return items


def flags_for(src: str, out: str, keep: List[str], token_limit: int, token_min: int, op: str) -> List[str]:
    from .vision import approx_tokens
    f = []
    low = out.lower()
    for w in keep:
        if w and w.lower() in src.lower() and w.lower() not in low:
            f.append(f"lost: {w}")
    for q in re.findall(r'"([^"]{2,80})"', src):
        if q.lower() not in low and op in ("convert", "rules", "refine"):
            f.append(f'quoted text changed: "{q[:30]}"')
    n = approx_tokens(out)
    if token_limit and n > token_limit:
        f.append(f"~{n} tokens > limit {token_limit}")
    if token_min and n < 0.5 * token_min:
        f.append(f"short: ~{n} tokens")
    if _PREAMBLE.match(out):
        f.append("starts with a preamble")
    if " ".join(src.split()).lower() == " ".join(out.split()).lower():
        f.append("unchanged")
    if op == "pair":                                  # a twin should differ in few words
        a, b = src.lower().split(), out.lower().split()
        import difflib
        changed = 1.0 - difflib.SequenceMatcher(None, a, b).ratio()
        if changed > 0.30:
            f.append(f"twin changed {round(100 * changed)}% of words")
    return f


class RewriteJob:
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
        items = [i for i in spec.get("items") or [] if (i.get("text") or "").strip()]
        if not items:
            raise ValueError("no prompts to rewrite")
        op = spec.get("op")
        if op not in OPS:
            raise ValueError("unknown operation")
        if op in ("edit", "pair") and not (spec.get("instruction") or "").strip():
            raise ValueError("write the change you want (instruction)")
        if op == "rules" and not (spec.get("rules") or "").strip():
            raise ValueError("write at least one rule")
        if op == "refine":
            missing = sum(1 for i in items if not (i.get("image") and os.path.isfile(i["image"])))
            if missing == len(items):
                raise ValueError("refine needs the source images - none of these prompts has a known image")
        n_var = max(1, min(10, int(spec.get("variations") or 3))) if op == "variations" else 1
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("a rewrite run is already going")
            self.state = {"running": True, "done": 0, "total": len(items) * n_var, "errors": 0, "last_error": "",
                          "results": [], "started": time.time()}
            self.cancel = threading.Event()
        threading.Thread(target=self._run, args=(settings, items, spec, op, n_var), daemon=True).start()
        return self.status()

    def _run(self, settings, items, spec, op, n_var) -> None:
        from .generate import extract_json
        from .instructions import compose, load_presets
        from .jobs import make_client, resolve_model
        from .model_caps import caps_for, normalise_mode
        from .vision import _image_url
        try:
            presets = load_presets(config.PRESETS_DIR)
            preset = presets[spec.get("preset") or settings["default_preset"]]
            composed = compose(preset, [], "", 1)
            client = make_client(settings)
            api_id, gguf, _ = resolve_model(client, spec.get("model") or "__loaded__")
            caps = caps_for(gguf)
            mode_r = normalise_mode(spec.get("reasoning") or preset.reasoning, caps)
            keep = [w.strip() for w in re.split(r"[,\n]", spec.get("keep") or "") if w.strip()]
            if spec.get("trigger"):
                keep.append(spec["trigger"].strip().rstrip(".,;: "))
            word = WORDS.get(spec.get("words") or "fresh", WORDS["fresh"])
            keep_rules = ("Keep these words and names exactly as written: " + ", ".join(keep) + "\n") if keep else ""
            keep_rules += "Keep any text in double quotes (text to be rendered in the image) exactly.\n"
            if preset.output == "json" and "ideogram" in preset.id:   # no picture to measure: do not invent a layout
                keep_rules += ("Bounding boxes: when no image is attached you cannot see where things are, so leave \"bbox\" "
                               "out of every element (Ideogram accepts elements without boxes) - except where the prompt "
                               "states a clear position (left, right, top, bottom, foreground, background, centre); then give "
                               "one rough box on the 0-1000 grid that follows that wording. When an image is attached, place "
                               "the boxes from the image.\n")
            temp = float(spec.get("temperature") or (0.9 if op == "variations" else 0.7))
            with self.lock:
                self.state["model"] = api_id
            done = 0
            for idx, it in enumerate(items, 1):
                earlier: List[str] = []
                for k in range(1, n_var + 1):
                    if self.cancel.is_set():
                        raise InterruptedError
                    try:
                        src = " ".join(it["text"].split())
                        img = it.get("image") if op == "refine" and it.get("image") and os.path.isfile(it["image"]) else ""
                        if op == "refine" and not img:
                            raise ValueError("no source image for this prompt")
                        facts = ""
                        if op == "refine" and img and spec.get("facts"):
                            try:
                                from .facts import cached_analyze, direction_text
                                ft = direction_text(cached_analyze(img))
                                facts = ("\nMeasured by a head-pose detector (more reliable than your impression for "
                                         "head direction): " + ft) if ft else ""
                            except Exception:
                                facts = ""
                        op_rules = OPS[op].format(instruction=spec.get("instruction", "").strip(),
                                                  rules=spec.get("rules", "").strip(), k=k, n=n_var,
                                                  earlier="; ".join(earlier) or "none yet", facts=facts)
                        if op == "pair":               # a twin keeps the input's own form - no target-format rewrite
                            system = RW_SYSTEM.format(op_rules=op_rules, word_rules=WORDS["faithful"],
                                                      keep_rules=keep_rules).split("The target model and its prompt")[0]
                        else:
                            system = RW_SYSTEM.format(op_rules=op_rules, word_rules=word, keep_rules=keep_rules) + \
                                "\n\n" + composed.system
                        user: Any = "INPUT PROMPT:\n" + src
                        if img:
                            user = [{"type": "text", "text": "INPUT PROMPT:\n" + src + "\n\nThe image it describes:"},
                                    {"type": "image_url", "image_url": {"url": _image_url(img)}}]
                        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                        extra = 4096 if mode_r not in ("off", "") and not mode_r.startswith("i") else 0
                        r = client.chat(api_id, msgs, max_tokens=preset.max_tokens + extra, temperature=temp,
                                        top_p=preset.top_p, reasoning=mode_r, caps=caps)
                        out = (r.content or "").strip()
                        if preset.output == "json":
                            out = extract_json(out)
                            if preset.layout:
                                from .layout import apply as apply_layout
                                _img = (it.get('image') or None)
                                _wh = None
                                if _img:
                                    from PIL import Image as _I
                                    with _I.open(_img) as _im:
                                        _wh = _im.size
                                out, _canvas = apply_layout(preset.layout, out, spec.get('layout'), _wh)
                                out = out + ''
                        if not out:
                            raise ValueError("empty answer")
                        if spec.get('trigger'):
                            from .instructions import apply_trigger
                            out = apply_trigger(out, spec['trigger'], preset.output)
                        if op == "variations":
                            earlier.append(" ".join(out.split()[:10]) + " ...")
                        res = {"n": len(self.state["results"]) + 1, "src_index": idx, "k": k, "src": src, "prompt": out,
                               "image": it.get("image", ""), "op": op, "preset": preset.id, "model": api_id,
                               "instruction": (spec.get("instruction") or spec.get("rules") or "").strip(),
                               "flags": (["graded duff before"] if _graded_duff(out) else []) + flags_for(src, out, keep, preset.token_limit,
                                                  0 if op == "pair" else preset.token_target[0], op)}
                        with self.lock:
                            self.state["results"].append(res)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        with self.lock:
                            self.state["errors"] += 1
                            self.state["last_error"] = f"prompt {idx}: {type(exc).__name__}: {exc}"
                    done += 1
                    with self.lock:
                        self.state["done"] = done
        except InterruptedError:
            pass
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.state.update(running=False, finished=time.time())


def save_pairs(settings: Dict[str, Any], results: List[Dict[str, Any]], name: str = "") -> str:
    """Two line-aligned lists: <name>_A.txt (originals) and <name>_B.txt (twins)."""
    from .outputs import unique_path
    if not results:
        raise ValueError("nothing to save")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    base = (name or "").strip() or f"pairs_{stamp}"
    base = base[:-4] if base.lower().endswith(".txt") else base
    a = unique_path((base if os.path.isabs(base) else os.path.join(settings["group_dir"], base)) + "_A.txt")
    b = a[:-6] + "_B.txt"
    if os.path.exists(b):
        raise ValueError(f"{b} already exists - choose another name")
    with open(a, "w", encoding="utf-8", newline="\n") as h:
        h.write("\n".join(" ".join(r["src"].split()) for r in results) + "\n")
    with open(b, "w", encoding="utf-8", newline="\n") as h:
        h.write("\n".join(" ".join(r["prompt"].split()) for r in results) + "\n")
    return a + "  +  " + os.path.basename(b)


def save_results(settings: Dict[str, Any], results: List[Dict[str, Any]], name: str = "") -> str:
    from .outputs import unique_path
    from .store import append_index
    if not results:
        raise ValueError("nothing to save")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    name = (name or "").strip() or f"rewrite_{stamp}"
    if not name.lower().endswith(".txt"):
        name += ".txt"
    path = unique_path(name if os.path.isabs(name) else os.path.join(settings["group_dir"], name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    made = _dt.datetime.now().isoformat(timespec="seconds")
    lines, meta, index = [], [], []
    for r in results:
        text = " ".join(r["prompt"].split())
        lines.append(text)
        meta.append({"line": len(lines), "variant": f"rewrite {r['op']}", "preset": r["preset"], "model": r["model"],
                     "source_prompt": r["src"], "image": r.get("image", ""), "flags": r.get("flags", []),
                     "prompt_sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]})
        index.append({"file": path, "mode": "rewrite", "image": r.get("image", ""), "prompt": text, "model": r["model"],
                      "preset": r["preset"], "instr": "", "variant": f"rewrite {r['op']}", "made": made})
    with open(path, "w", encoding="utf-8", newline="\n") as h:
        h.write("\n".join(lines) + "\n")
    with open(path[:-4] + ".meta.jsonl", "w", encoding="utf-8", newline="\n") as h:
        for rec in meta:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    append_index(config.RUNS_DIR, index)
    return path


JOB = RewriteJob()


def _graded_duff(text: str) -> bool:
    from .prompt_grades import grade_of
    try:
        return grade_of(text) == "duff"
    except Exception:
        return False
