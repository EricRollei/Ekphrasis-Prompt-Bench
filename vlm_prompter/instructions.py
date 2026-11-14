"""
Layered instructions: Preset (target format) + Modifiers (run-wide) + per-image Note -> system + user messages.

presets/*.yaml   one preset per file (see presets/krea2.yaml for every field)
modifiers/*.yaml each file is a list of {id, name, text}; text may contain {value} (filled per use)

The composed text is hashed; the hash goes into the cache key and the run log, so every result ties back to the
exact instructions that produced it.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class Preset:
    id: str
    name: str
    target: str
    system: str
    user: str
    token_target: Tuple[int, int] = (400, 480)
    token_limit: int = 0                  # target model's encoder limit (0 = none); outputs above it are flagged
    max_tokens: int = 1200
    reasoning: str = "off"                # off | on | auto
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    notes: str = ""
    output: str = "text"                  # text | json  (json: answer must be a JSON object/array)
    layout: str = ""                      # "" | moss: bbox layout preset - boxes converted for a canvas size
    path: str = ""


@dataclass
class Modifier:
    id: str
    name: str
    text: str

    @property
    def takes_value(self) -> bool:
        return "{value}" in self.text


@dataclass
class Composed:
    system: str
    user: str
    variations: int = 1
    sha: str = ""
    parts: Dict[str, object] = field(default_factory=dict)


def load_presets(folder: str) -> Dict[str, Preset]:
    presets: Dict[str, Preset] = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        pid = data.get("id") or os.path.splitext(os.path.basename(path))[0]
        tt = data.get("token_target") or [400, 480]
        presets[pid] = Preset(
            id=pid, name=data.get("name", pid), target=data.get("target", pid),
            system=(data.get("system") or "").strip(), user=(data.get("user") or "").strip(),
            token_target=(int(tt[0]), int(tt[1])), token_limit=int(data.get("token_limit") or 0),
            max_tokens=int(data.get("max_tokens") or 1200), reasoning=data.get("reasoning", "off"),
            temperature=float(data.get("temperature", 0.7)),
            top_p=(float(data["top_p"]) if data.get("top_p") is not None else None),
            top_k=(int(data["top_k"]) if data.get("top_k") is not None else None),
            min_p=(float(data["min_p"]) if data.get("min_p") is not None else None),
            presence_penalty=(float(data["presence_penalty"]) if data.get("presence_penalty") is not None else None),
            repeat_penalty=(float(data["repeat_penalty"]) if data.get("repeat_penalty") is not None else None),
            notes=data.get("notes", "") or "", output=str(data.get("output", "text")).lower(), layout=str(data.get("layout") or ""), path=path,
        )
    return presets


def load_modifiers(folder: str) -> Dict[str, Modifier]:
    mods: Dict[str, Modifier] = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or []
        for item in data:
            mods[item["id"]] = Modifier(id=item["id"], name=item.get("name", item["id"]), text=item["text"].strip())
    return mods


def parse_modifier_args(args: List[str], available: Dict[str, Modifier]) -> List[Tuple[Modifier, str]]:
    """CLI form: id or id=value. Unknown ids raise with the list of known ones."""
    chosen = []
    for arg in args or []:
        mid, _, value = arg.partition("=")
        mid = mid.strip()
        if mid not in available:
            raise KeyError(f"unknown modifier {mid!r}; known: {', '.join(sorted(available))}")
        mod = available[mid]
        if mod.takes_value and not value.strip():
            raise ValueError(f"modifier {mid!r} needs a value: --mod {mid}=\"...\"")
        chosen.append((mod, value.strip()))
    return chosen


VARIATIONS_BLOCK = (
    "Write {n} different prompts for this image, numbered exactly like this on their own lines: "
    "\"### 1\", \"### 2\" ... Each prompt keeps the same main subject and what makes the image itself, but each one "
    "after the first must clearly differ from the earlier ones in lighting, camera position and lens, colour palette, "
    "mood and secondary details - choose less obvious options each time rather than the first idea that comes to mind. "
    "Each prompt follows all the rules above and has the full length."
)


def compose(preset: Preset, modifiers: List[Tuple[Modifier, str]] = None, note: str = "",
            variations: int = 1) -> Composed:
    lo, hi = preset.token_target
    fmt = {"token_min": lo, "token_max": hi, "words_min": int(lo * 0.75), "words_max": int(hi * 0.75)}
    system = preset.system.format(**fmt)
    user = preset.user.format(**fmt)

    extra = []
    for mod, value in modifiers or []:
        extra.append(mod.text.replace("{value}", value))
    if note.strip():
        extra.append(note.strip())
    if extra:
        user += "\n\nAdditional instructions (these override the image where they conflict):\n" + "\n".join(
            f"- {e}" for e in extra)
    if variations > 1:
        user += "\n\n" + VARIATIONS_BLOCK.format(n=variations)

    sha = hashlib.sha1((system + "\x00" + user).encode("utf-8")).hexdigest()[:16]
    return Composed(system=system, user=user, variations=variations, sha=sha, parts={
        "preset": preset.id, "modifiers": [(m.id, v) for m, v in modifiers or []], "note": note,
    })
