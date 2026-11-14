"""
What reasoning controls a model supports, read from the chat template stored in its GGUF.

  reason_tag  template parses "{REASON:...}" from user messages (DavidAU TWIN-TURBO: 5 thinking + 5 instruct modes)
  effort      template uses reasoning_effort (Qwen3.8: low / medium / xhigh)
  thinking    template has a thinking on/off switch (enable_thinking or <think> blocks)

Results are cached in .cache/model_caps.json keyed by path + size + mtime, so each GGUF is read once.
Reading uses the gguf package (metadata only; the weights are memory-mapped, not loaded).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Tuple

from . import config

_CACHE_PATH = os.path.join(config.CACHE_DIR, "model_caps.json")
_LOCK = threading.Lock()

MODE_LABELS = {
    "off": "no thinking",
    "on": "thinking (model default)",
    "imedium": "instruct, medium (no thinking)",
    "ilow": "instruct, low (no thinking)",
    "ixhigh": "instruct, xhigh (no thinking)",
    "ieinstein": "instruct, einstein (no thinking, adds expert-panel persona)",
    "ispoon": "instruct, spoon (no thinking, research persona)",
    "low": "thinking, low",
    "medium": "thinking, medium",
    "xhigh": "thinking, xhigh (model default)",
    "einstein": "thinking, einstein (slow)",
    "spoon": "thinking, spoon (very slow: 5000+ word reasoning)",
}


def _read_template(path: str) -> str:
    from gguf import GGUFReader                      # imported lazily: only needed when a new GGUF is inspected
    reader = GGUFReader(path)
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        return ""
    return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")


def tag_usable(tpl: str) -> bool:
    """Does the template really parse "{REASON:...}"? Renders a tagged message with jinja2 (LM Studio's engine is
    stricter: an undefined variable in an `in` test is an error there, a silent False in jinja2) and checks the tag
    was consumed. Found Nov 14 2026: DavidAU Qwen3.8 709-L tests `candidate_key` without ever setting it, so every
    tagged request failed with 'Cannot perform operation in on undefined values'."""
    try:
        import jinja2
    except ImportError:                                   # heuristic for the known defect
        return not ("candidate_key" in tpl and "set candidate_key" not in tpl)
    env = jinja2.Environment()

    def _raise(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = _raise
    try:
        out = env.from_string(tpl).render(messages=[{"role": "user", "content": "{REASON:xhigh} hello"}],
                                          add_generation_prompt=True)
    except Exception:
        return False
    if "{REASON:" in out:
        return False
    return not ("candidate_key" in tpl and "set candidate_key" not in tpl)


def caps_for(path: str) -> Dict[str, bool]:
    """{"reason_tag", "effort", "thinking", "known"} for a GGUF path. Unknown/unreadable -> thinking on/off only."""
    unknown = {"reason_tag": False, "effort": False, "thinking": True, "known": False}
    if not path or not os.path.isfile(path):
        return unknown
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}|v2"            # v2: tag_broken check added Nov 14
    with _LOCK:
        cache = {}
        if os.path.isfile(_CACHE_PATH):
            try:
                with open(_CACHE_PATH, "r", encoding="utf-8") as handle:
                    cache = json.load(handle)
            except (OSError, json.JSONDecodeError):
                cache = {}
        if key in cache:
            return cache[key]
    try:
        tpl = _read_template(path)
    except Exception as exc:  # gguf missing or unreadable file: fall back, never fail a run over this
        print(f"WARNING: could not read chat template of {path}: {exc}")
        return unknown
    has_tag = "{REASON:" in tpl
    usable = has_tag and tag_usable(tpl)
    caps = {
        "reason_tag": usable,
        "tag_broken": has_tag and not usable,
        "effort": "reasoning_effort" in tpl,
        "thinking": "enable_thinking" in tpl or "<think>" in tpl,
        "known": True,
    }
    with _LOCK:
        cache[key] = caps
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=1)
    return caps


def modes_for(caps: Dict[str, bool]) -> List[Tuple[str, str]]:
    """(value, label) pairs for the reasoning dropdown, default first."""
    if caps.get("reason_tag"):
        order = ["imedium", "ilow", "ixhigh", "ieinstein", "ispoon", "low", "medium", "xhigh", "einstein", "spoon"]
    elif caps.get("effort"):
        order = ["off", "low", "medium", "xhigh"]
    elif caps.get("thinking"):
        order = ["off", "on"]
    else:
        order = ["off"]
    return [(m, MODE_LABELS[m]) for m in order]


def normalise_mode(mode: str, caps: Dict[str, bool]) -> str:
    """Map a preset's generic mode (off/on) or a mode from another model family to one this model supports."""
    valid = [m for m, _ in modes_for(caps)]
    if mode in valid:
        return mode
    if mode in ("", "off") or mode.startswith("i"):
        return valid[0]                       # first entry is always the no-thinking default
    if caps.get("effort"):
        return mode if mode in ("low", "medium", "xhigh") else "xhigh"
    return "on" if "on" in valid else valid[0]
