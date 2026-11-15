"""
Wildcards for the Ideas tab - a thin adapter around the vendored wildcard engine (vlm_prompter/wildcards, copied
from video_prompter core/wildcards; refresh it with tools/sync_wildcards.py).

The engine brings the full grammar (Dynamic Prompts compatible plus extensions), the per-folder catalog
(_catalog.sqlite: display names, kinds, facets, samples, NSFW levels) and the user's curation (_catalog_user.yaml:
stars, hidden lists, dropped entries) - shared with video_prompter when both point at the same folder.

Defaults come from settings (Settings page): wildcards_dir, wc_nsfw, wc_violence, wc_content_filter, wc_strip_lora.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import os
import random
import re
import threading
from typing import Any, Callable, Dict, List, Optional

from . import config
from .wildcards import NSFW_LEVELS, VIOLENCE_LEVELS, WildcardEngine

CACHE_DIR = os.path.join(config.APP_DIR, ".cache", "wildcard_catalogs")
_LOCK = threading.Lock()
MIN_ENTRIES = 8                 # Eric's triage rule: smaller lists are rarely useful as wildcards
_BAD_KEY = re.compile(r"[<>\[\],{}|$]")    # keys that cannot be written as __key__ (structured-file fragments)
_ENGINES: Dict[tuple, WildcardEngine] = {}

LLM_SLOT_SYSTEM = ("You fill ONE slot in an image prompt. Answer with only the requested words - a short phrase of "
                   "1 to 8 words, no quotes, no explanation, no full sentence. Be specific and visual, and avoid the "
                   "most obvious answer.")


def options(settings: Dict[str, Any]) -> Dict[str, Any]:
    nsfw = settings.get("wc_nsfw", "off")
    violence = settings.get("wc_violence", "off")
    return {"dir": settings.get("wildcards_dir") or "",
            "nsfw": nsfw if nsfw in NSFW_LEVELS else "off",
            "violence": violence if violence in VIOLENCE_LEVELS else "off",
            "content_filter": bool(settings.get("wc_content_filter", True)),
            "strip_lora": bool(settings.get("wc_strip_lora", True))}


def engine(settings: Dict[str, Any]) -> WildcardEngine:
    o = options(settings)
    if not o["dir"] or not os.path.isdir(o["dir"]):
        raise ValueError(f"wildcards folder not found: {o['dir'] or '(not set)'} - set it in Settings")
    key = (os.path.normcase(os.path.abspath(o["dir"])), o["content_filter"], o["strip_lora"])
    with _LOCK:
        eng = _ENGINES.get(key)
        if eng is None:
            eng = WildcardEngine(extra_paths=[o["dir"]], include_builtin=False, include_configured=False,
                                 content_filter=o["content_filter"], strip_lora=o["strip_lora"])
            os.makedirs(CACHE_DIR, exist_ok=True)
            eng.enable_curation(CACHE_DIR, build_in_background=True)
            _ENGINES[key] = eng
    return eng


def _level(name: str) -> int:
    """Catalog ratings are sfw | suggestive | nudity | explicit; the engine's levels start at off."""
    from .wildcards.library import nsfw_level
    try:
        return nsfw_level({"sfw": "off", "": "off", None: "off"}.get(name, name))
    except Exception:
        return 0


def _row_dict(eng: WildcardEngine, row) -> Dict[str, Any]:
    d = row.to_dict()
    d["display"] = (d.get("display") or row.key).replace("\u00c2\u00b7", "\u00b7").replace("\u00c2", "")
    cur = eng.curation
    d["stars"] = cur.stars(row.key) if cur else 0
    d["samples"] = (d.get("samples") or [])[:8]
    return d


def search(settings: Dict[str, Any], query: str = "", facet: str = "", limit: int = 80,
           small: bool = False) -> Dict[str, Any]:
    """Lists for the browser: catalog search (or best lists when the query is empty), hidden lists and lists above
    the NSFW level left out, starred and better lists first."""
    eng = engine(settings)
    o = options(settings)
    if eng.catalog is None:
        return {"rows": [], "facets": [], "total": 0, "note": "catalog not available"}
    from .wildcards.library import nsfw_level
    cap = nsfw_level(o["nsfw"])
    rows_all = eng.catalog.rows()
    if query.strip():
        found = eng.catalog.search(query.strip(), limit=600)
    else:
        found = list(rows_all.values())
    cur = eng.curation
    out = []
    for row in found:
        if facet and row.facet != facet:
            continue
        if cur and cur.hidden(row.key):
            continue
        if _BAD_KEY.search(row.key):
            continue
        if not small and row.entries < MIN_ENTRIES:
            continue
        if row.kind in ("template",) and not query.strip():
            continue
        if _level(row.nsfw) > cap:
            continue
        out.append(row)
    # starred lists first, then the catalog's quality (it already ranks text relevance within a query)
    order = {r.key: i for i, r in enumerate(out)}
    out.sort(key=lambda r: (-(cur.stars(r.key) if cur else 0), -round(r.quality, 1), order[r.key]))
    facets: Dict[str, int] = {}
    for r in rows_all.values():
        facets[r.facet] = facets.get(r.facet, 0) + 1
    return {"rows": [_row_dict(eng, r) for r in out[:limit]], "total": len(out),
            "facets": sorted(facets.items(), key=lambda x: -x[1]), "nsfw": o["nsfw"], "lists": len(rows_all)}


def entries(settings: Dict[str, Any], key: str, limit: int = 400) -> Dict[str, Any]:
    eng = engine(settings)
    o = options(settings)
    items = eng.library.entries_of(key, nsfw=o["nsfw"])
    texts = [e.text for e in items]
    return {"key": key, "count": len(texts), "entries": texts[:limit], "truncated": len(texts) > limit,
            "stars": eng.curation.stars(key) if eng.curation else 0}


def make_llm_provider(client, api_id: str, caps: Dict[str, Any]) -> Callable[[str, str, random.Random], Optional[str]]:
    """__llm:a 1980s hairstyle__ -> one short phrase from the loaded model, with the prompt so far as context."""
    def fill(argument: str, context: str, rng: random.Random) -> Optional[str]:
        msgs = [{"role": "system", "content": LLM_SLOT_SYSTEM},
                {"role": "user", "content": f"Slot: {argument.strip()}\nPrompt so far: {context[-600:].strip() or '(nothing yet)'}"}]
        r = client.chat(api_id, msgs, max_tokens=48, temperature=1.0, seed=rng.randrange(1 << 30), reasoning="off",
                        caps=caps)
        text = (r.content or "").strip().strip('"').strip()
        return text.splitlines()[0][:160] if text else None
    return fill


def process(settings: Dict[str, Any], template: str, seed: Optional[int] = None,
            providers: Optional[Dict[str, Any]] = None):
    o = options(settings)
    return engine(settings).process(template, seed=seed, nsfw=o["nsfw"], violence=o["violence"],
                                    providers=providers or {})


def roll(settings: Dict[str, Any], template: str, n: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
    """Raw fills without the model (llm slots stay as their instruction text)."""
    rng = random.Random(seed)
    fills = []
    for _ in range(max(1, min(20, n))):
        res = process(settings, template, seed=rng.randrange(0, 0x7FFFFFFF))
        fills.append({"text": res.text, "seed": res.seed, "blocked": res.blocked,
                      "picks": [f"{s.label or s.source}: {s.value}" for s in res.top_level_slots()],
                      "warnings": [w for w in res.warnings][:6]})
    return {"fills": fills}
