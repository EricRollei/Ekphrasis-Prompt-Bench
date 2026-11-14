"""
Writers for the three output modes, group naming, and the no-overwrite rule.

Modes (any combination):
  per-image   <stem>.prompt.txt beside the image (never <stem>.txt - training captions live there)
  per-folder  one file per group in settings["group_dir"], named <anchor>_<key>_prompts.txt (SPEC 6a)
  combined    one file for the whole job in settings["group_dir"] (or an explicit path)

No-overwrite: an existing target becomes name.2.txt, name.3.txt ... - nothing is ever replaced.
Formats for group/combined files: "lines" (one prompt per line - Krea2 Sweep Plan / wildcard ready) or "blocks"
(# relative/path header, prompt, blank line).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_ID_RE = re.compile(r"^(\d{4,})(?=[-_ .]|$)")


# ----------------------------------------------------------------------------------------------------- text helpers
def safe_rel(path: str, base: str) -> str:
    """Path relative to base, or the full path when that is impossible (e.g. a different drive)."""
    if not base:
        return path
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique_path(path: str) -> str:
    """path if free, else name.2.ext, name.3.ext ... (the full suffix like .prompt.txt stays together)."""
    if not os.path.exists(path):
        return path
    folder, name = os.path.split(path)
    stem, ext = name, ""
    for full_ext in (".prompt.txt", ".prompt.json", ".txt", ".jsonl"):
        if name.lower().endswith(full_ext):
            stem, ext = name[: -len(full_ext)], name[-len(full_ext):]
            break
    else:
        stem, ext = os.path.splitext(name)
    n = 2
    while True:
        candidate = os.path.join(folder, f"{stem}.{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def write_text_new(path: str, text: str) -> str:
    """Write to a fresh path (never overwrite). Returns the path actually written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    target = unique_path(path)
    with open(target, "x", encoding="utf-8", newline="\n") as handle:    # "x" = fail rather than overwrite
        handle.write(text)
    return target


# ----------------------------------------------------------------------------------------------------- grouping
def group_folder(image_path: str, root: str, group_by: str, generic: set) -> str:
    """
    The folder whose file this image goes into.
      folder  -> the image's folder; generic folders (images, samples ...) merge upward, never above root
      depth:N -> root + the first N folders below it (shallower images stay in their own folder)
    """
    folder = os.path.dirname(os.path.abspath(image_path))
    root = os.path.abspath(root)
    if group_by.startswith("depth:"):
        depth = max(1, int(group_by.split(":", 1)[1]))
        rel = os.path.relpath(folder, root)
        if rel == ".":
            return root
        parts = rel.split(os.sep)
        return os.path.join(root, *parts[:depth])
    while os.path.normcase(folder) != os.path.normcase(root) and os.path.basename(folder).lower() in generic:
        folder = os.path.dirname(folder)
    return folder


def shorten_part(part: str, stop_words: set, max_len: int) -> str:
    """'3012122-JANUARY_ROMANOV' -> '3012122'; 'women_angaihi-mumbai' -> 'angaihi-mumbai'."""
    m = _ID_RE.match(part)
    if m:
        return m.group(1)
    kept_tokens = []
    for token in re.split(r"[_\s]+", part):
        sub = [s for s in token.split("-") if s and s.lower() not in stop_words]
        if sub:
            kept_tokens.append("-".join(sub))
    text = "_".join(kept_tokens) or part
    text = _ILLEGAL.sub("", text).strip(" ._-")
    return text[:max_len].rstrip(" ._-") or "folder"


def _non_generic_parts(folder: str, root: str, generic: set) -> List[str]:
    rel = os.path.relpath(folder, root)
    if rel == ".":
        return []
    return [p for p in rel.split(os.sep) if p.lower() not in generic]


def group_names(groups: Sequence[str], root: str, settings: Dict) -> Dict[str, str]:
    """
    Name stem (without _prompts.txt) for each group folder. anchor = first folder under root, key = deepest
    non-generic folder. Collisions within the job add the next ancestor, then a counter. GUI overrides win.
    """
    sep = settings.get("name_separator", "_")
    generic = {g.lower() for g in settings.get("generic_folders", [])}
    stop = {s.lower() for s in settings.get("stop_words", [])}
    max_len = int(settings.get("name_part_max", 24))
    overrides = settings.get("group_name_overrides", {}) or {}

    def short(p: str) -> str:
        return shorten_part(p, stop, max_len)

    def candidate(folder: str, extra_levels: int) -> str:
        parts = _non_generic_parts(folder, root, generic)
        if not parts:                                     # images directly in the start folder: name it after
            tail = [p for p in os.path.abspath(root).split(os.sep)   # its last two non-generic path parts
                    if p and not p.endswith(":") and p.lower() not in generic][-2:]
            return sep.join(short(p) for p in tail) or "root"
        if len(parts) == 1:
            return short(parts[0])
        middle = parts[1:-1][-extra_levels:] if extra_levels else []
        names = [short(parts[0])] + [short(p) for p in middle] + [short(parts[-1])]
        out = []
        for n in names:                                   # drop repeats (anchor == key after shortening)
            if not out or out[-1] != n:
                out.append(n)
        return sep.join(out)

    names: Dict[str, str] = {}
    levels = {g: 0 for g in groups}
    for _ in range(8):                                    # widen colliding names a level at a time
        names = {g: overrides.get(g) or candidate(g, levels[g]) for g in groups}
        seen: Dict[str, List[str]] = {}
        for g, n in names.items():
            seen.setdefault(n.lower(), []).append(g)
        clashes = [gs for gs in seen.values() if len(gs) > 1]
        if not clashes:
            return names
        for gs in clashes:
            for g in gs:
                if g not in overrides:
                    levels[g] += 1
    counts: Dict[str, int] = {}                          # still clashing: number them
    for g in groups:
        n = names[g]
        counts[n.lower()] = counts.get(n.lower(), 0) + 1
        if counts[n.lower()] > 1:
            names[g] = f"{n}{sep}{counts[n.lower()]}"
    return names


# ----------------------------------------------------------------------------------------------------- writers
def is_json_prompt(p: str) -> bool:
    s = p.lstrip()
    if not s.startswith(("{", "[")):
        return False
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


META_KEYS = ("model", "preset", "instr", "at")


def format_entries(entries: Iterable[Tuple[str, List[str]]], fmt: str, base: str, meta: Optional[Dict] = None) -> str:
    """fmt: lines (one prompt per line) | blocks (# path header + prompt) | jsonl ({"image", "prompt", model ...} per line)."""
    meta = meta or {}
    out = []
    for image, prompts in entries:
        rel = safe_rel(image, base)
        for i, p in enumerate(prompts, 1):
            if fmt == "blocks":
                head = f"# {rel}" + (f"  [{i}/{len(prompts)}]" if len(prompts) > 1 else "")
                out.append(f"{head}\n{p.strip()}\n")
            elif fmt == "jsonl":
                value = json.loads(p) if is_json_prompt(p) else one_line(p)
                row = {"image": rel, "prompt": value}
                row.update({k: v for k, v in (meta.get(image) or {}).items() if k in META_KEYS and v})
                out.append(json.dumps(row, ensure_ascii=False))
            else:
                out.append(one_line(p) if not is_json_prompt(p) else json.dumps(json.loads(p), ensure_ascii=False))
    return ("\n".join(out)).rstrip("\n") + "\n" if out else ""


def existing_prompts(path: str) -> set:
    """Prompts already in a list file (lines, blocks or jsonl), normalised for duplicate checks."""
    found = set()
    if not os.path.isfile(path):
        return found
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            s = line.strip()
            if not s or s.startswith("# "):
                continue
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and "prompt" in obj:
                        v = obj["prompt"]
                        found.add(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else one_line(v))
                        continue
                except ValueError:
                    pass
            found.add(one_line(s))
    return found


def _norm(p: str) -> str:
    return json.dumps(json.loads(p), ensure_ascii=False) if is_json_prompt(p) else one_line(p)


def prompt_sha(p: str) -> str:
    return hashlib.sha1(_norm(p).lower().encode("utf-8")).hexdigest()[:16]


def count_prompts(path: str, fmt: str) -> int:
    """How many prompts a list file already holds (blocks: one per '# ' header; lines/jsonl: one per line)."""
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            s = line.strip()
            if fmt == "blocks":
                n += s.startswith("# ")
            elif s and not s.startswith("# "):
                n += 1
    return n


def sidecar_path(list_path: str) -> str:
    stem = list_path
    for ext in (".jsonl", ".txt"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem + ".meta.jsonl"


def write_list(path: str, entries: List[Tuple[str, List[str]]], fmt: str, base: str, on_exists: str = "new",
               dedupe: bool = False, meta: Optional[Dict] = None, sidecar: bool = False) -> Dict:
    """
    on_exists "new"    -> an existing file is left alone; we write name.2.txt ...
              "append" -> new prompts are added to the end of the existing file (nothing in it is changed)
    dedupe: drop prompts already present in the target file (append) or repeated within this batch.
    sidecar: also write/append name.meta.jsonl - one line per prompt written, in the same order:
             {"n": position in the list (1-based), "prompt_sha", "image", model, preset, instr, at}
    Returns {path, action, count, sidecar, _written: [(image, prompt)]} - _written is for the index, not the page.
    """
    meta = meta or {}
    seen = existing_prompts(path) if (dedupe and on_exists == "append") else set()
    if dedupe:
        filtered = []
        for image, prompts in entries:
            keep = []
            for p in prompts:
                n = _norm(p)
                if n not in seen:
                    seen.add(n)
                    keep.append(p)
            if keep:
                filtered.append((image, keep))
        entries = filtered
    flat = [(image, p) for image, ps in entries for p in ps]
    if not flat:
        return {"path": path, "action": "unchanged", "count": 0, "sidecar": None, "_written": []}
    text = format_entries(entries, fmt, base, meta)
    if on_exists == "append" and os.path.isfile(path):
        start = count_prompts(path, fmt)
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            needs_newline = False
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) not in (b"\n",)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(("\n" if needs_newline else "") + text)
        target, action = path, "appended"
    else:
        start = 0
        target, action = write_text_new(path, text), "new"
    side = None
    if sidecar and fmt != "jsonl":                 # .jsonl lists already carry model/instructions on every line
        side = sidecar_path(target)
        rel = lambda img: safe_rel(img, base)
        with open(side, "a", encoding="utf-8", newline="\n") as handle:
            for k, (image, p) in enumerate(flat, start + 1):
                row = {"n": k, "prompt_sha": prompt_sha(p), "image": rel(image)}
                row.update({m: v for m, v in (meta.get(image) or {}).items() if m in META_KEYS and v})
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"path": target, "action": action, "count": len(flat), "sidecar": side, "_written": flat}


def write_per_image(image: str, prompts: List[str], suffix: str, json_suffix: str = ".prompt.json") -> str:
    stem = os.path.splitext(image)[0]
    if prompts and all(is_json_prompt(p) for p in prompts):
        objs = [json.loads(p) for p in prompts]
        body = json.dumps(objs[0] if len(objs) == 1 else objs, ensure_ascii=False, indent=2) + "\n"
        return write_text_new(stem + json_suffix, body)
    body = "\n".join(one_line(p) for p in prompts) + "\n"
    return write_text_new(stem + suffix, body)


def per_image_exists(image: str, suffix: str, json_suffix: str = ".prompt.json") -> bool:
    stem = os.path.splitext(image)[0]
    return os.path.exists(stem + suffix) or os.path.exists(stem + json_suffix)


def plan_groups(items: Sequence[Tuple[str, str]], settings: Dict, group_by: str = "folder"
                ) -> "OrderedDict[Tuple[str, str], List[str]]":
    """(root, group_folder) -> [images] in input order."""
    generic = {g.lower() for g in settings.get("generic_folders", [])}
    groups: "OrderedDict[Tuple[str, str], List[str]]" = OrderedDict()
    for image, root in items:
        g = group_folder(image, root, group_by, generic)
        groups.setdefault((root, g), []).append(image)
    return groups


def group_file_names(groups: "OrderedDict[Tuple[str, str], List[str]]", settings: Dict) -> Dict[Tuple[str, str], str]:
    """(root, group) -> file name (no folder). Names are made unique across roots too."""
    by_root: Dict[str, List[str]] = OrderedDict()
    for root, g in groups:
        by_root.setdefault(root, []).append(g)
    result: Dict[Tuple[str, str], str] = {}
    used: Dict[str, int] = {}
    sep = settings.get("name_separator", "_")
    for root, folders in by_root.items():
        for g, stem in group_names(folders, root, settings).items():
            low = stem.lower()
            used[low] = used.get(low, 0) + 1
            if used[low] > 1:
                stem = f"{stem}{sep}{used[low]}"
            result[(root, g)] = f"{stem}{sep}prompts.txt"
    return result


def combined_name(items: Sequence[Tuple[str, str]], settings: Dict, ext: str = ".txt") -> str:
    sep = settings.get("name_separator", "_")
    stop = {s.lower() for s in settings.get("stop_words", [])}
    roots = list(OrderedDict.fromkeys(r for _, r in items))
    stems = [shorten_part(os.path.basename(r.rstrip("\\/")) or "root", stop, int(settings.get("name_part_max", 24)))
             for r in roots[:3]]
    stem = sep.join(stems) + (f"{sep}more" if len(roots) > 3 else "")
    return f"{stem}{sep}combined{sep}prompts{ext}"


def write_caption(image: str, prompt: str) -> Dict[str, Any]:
    """Training caption <stem>.txt beside the image (ai-toolkit / kohya / OneTrainer read these).
    Never overwrites: an existing caption is left alone and reported as skipped."""
    path = os.path.splitext(image)[0] + ".txt"
    if os.path.exists(path):
        return {"path": path, "action": "skipped", "count": 0}
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(one_line(prompt) + "\n")
    return {"path": path, "action": "new", "count": 1}
