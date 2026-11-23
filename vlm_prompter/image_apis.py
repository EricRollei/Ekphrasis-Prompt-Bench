"""
Hosted image-generation backends for rendering (Phase 2 of private_docs/SPEC_render.md).

Every model has a capability description ("caps"): how its size is chosen (aspect ratio and/or resolution tier,
fixed size presets, or width x height), whether it takes a seed, its quality levels and its own options.
  - Black Forest Labs, OpenAI, Ideogram, Google and xAI: written here from each provider's docs (checked 2026-11-22).
  - fal.ai and Replicate: read live from the model's own input schema (OpenAPI), so any of their models works.
The render preset stores aspect_ratio / resolution / width / height / quality / extra (model options, JSON).

API keys live only in the local settings.json (api_keys); they are never sent to the page or committed.

Request shapes:
  BFL        POST https://api.bfl.ai/v1/{model} (x-key) -> poll polling_url -> result.sample
  OpenAI     POST https://api.openai.com/v1/images/generations (Bearer) -> data[].b64_json
  Ideogram   POST https://api.ideogram.ai/v2/image/generate/{model} (Api-Key) -> data[].url
  Google     POST https://generativelanguage.googleapis.com/v1beta/interactions (x-goog-api-key) -> image blocks
  xAI        POST https://api.x.ai/v1/images/generations (Bearer) -> data[].b64_json / url
  fal.ai     POST https://queue.fal.run/{model} (Key) -> poll status_url -> GET response_url -> images[].url
             schema: https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={model}
  Replicate  POST https://api.replicate.com/v1/models/{owner}/{name}/predictions (Bearer, Prefer: wait)
             or /v1/predictions with "version" for owner/name:version -> poll urls.get -> output URL(s)
             schema: GET /v1/models/{owner}/{name} -> latest_version.openapi_schema.components.schemas.Input

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

R15 = ["21:9", "2:1", "16:9", "3:2", "7:5", "4:3", "5:4", "1:1", "4:5", "3:4", "5:7", "2:3", "9:16", "1:2", "9:21"]
R11 = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]
R_ULTRA = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"]
R_GOOGLE = ["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
R_XAI = ["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]

_SAFETY = {"type": "int", "min": 0, "max": 7, "default": 2, "label": "safety tolerance (0 strictest - 7)"}


def _cap(size: str, **kw) -> Dict[str, Any]:
    return {"size": size, "ratios": [], "resolutions": [], "presets": [], "seed": True, "qualities": [], "options": {},
            "note": "", **kw}


MODELS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "bfl": {
        "flux-3-image": _cap("ratio_res", ratios=["auto"] + R15, resolutions=["768sq", "1k", "1.5k", "2k", "4k"], seed=False,
                             options={"safety_tolerance": _SAFETY,
                                      "grounding": {"type": "bool", "default": True, "label": "grounding (web research for the prompt)"}},
                             note="FLUX 3 Image takes no seed; 'auto' lets the prompt choose the framing."),
        "flux-3-image-klein": _cap("ratio_res", ratios=["auto"] + R15, resolutions=["768sq", "1k", "1.5k", "2k", "4k"],
                                   options={"safety_tolerance": _SAFETY,
                                            "steps": {"type": "int", "min": 1, "max": 16, "default": 16, "label": "steps"}}),
        "flux-2-pro-preview": _cap("ratio_mp", ratios=R11, resolutions=["1MP", "2MP", "4MP"], multiple=16),
        "flux-2-pro": _cap("ratio_mp", ratios=R11, resolutions=["1MP", "2MP", "4MP"], multiple=16),
        "flux-2-max": _cap("ratio_mp", ratios=R11, resolutions=["1MP", "2MP", "4MP"], multiple=16),
        "flux-2-flex": _cap("ratio_mp", ratios=R11, resolutions=["1MP", "2MP", "4MP"], multiple=16),
        "flux-2-klein-9b": _cap("ratio_mp", ratios=R11, resolutions=["1MP", "2MP", "4MP"], multiple=16),
        "flux-2-klein-4b": _cap("ratio_mp", ratios=R11, resolutions=["1MP", "2MP", "4MP"], multiple=16),
        "flux-pro-1.1-ultra": _cap("ratio", ratios=R_ULTRA,
                                   options={"raw": {"type": "bool", "default": False, "label": "raw (candid photographic look)"}}),
        "flux-pro-1.1": _cap("ratio_mp", ratios=R11, resolutions=["1MP"], multiple=32, max_side=1440),
    },
    "openai": {
        m: _cap("preset", presets=["1024x1024", "1536x1024", "1024x1536", "auto"], seed=False, qualities=["auto", "low", "medium", "high"])
        for m in ("gpt-image-2.6-flare", "gpt-image-2.6-sunburst")
    },
    "ideogram": {
        m: _cap("ratio_res", ratios=R11, resolutions=["1K", "2K"], qualities=["high", "medium", "low"],
                options={"magic_prompt": {"type": "enum", "values": ["auto", "on", "off"], "default": "off",
                                          "label": "magic prompt (off keeps your wording)"}})
        for m in ("ideogram-4-5", "ideogram-4-6", "ideogram-4", "ideogram-3")
    },
    "google": {
        "gemini-nano-banana-2.1": _cap("ratio_res", ratios=R_GOOGLE, resolutions=["1K", "2K", "4K"], seed=False),
        "gemini-3-pro-image": _cap("ratio_res", ratios=R_GOOGLE, resolutions=["1K", "2K", "4K"], seed=False),
        "gemini-3.1-flash-image": _cap("ratio_res", ratios=R_GOOGLE, resolutions=["512", "1K", "2K", "4K"], seed=False),
        "gemini-3.1-flash-lite-image": _cap("ratio_res", ratios=R_GOOGLE, resolutions=["1K"], seed=False),
    },
    "xai": {
        "grok-imagine-image-2.0": _cap("ratio_res", ratios=R_XAI, resolutions=["1k", "1.5k", "2k"], seed=False,
                                       qualities=["auto", "low", "medium"]),
    },
}

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "bfl": {"label": "Black Forest Labs (FLUX)", "key_hint": "BFL API key (dashboard.bfl.ai)"},
    "openai": {"label": "OpenAI (GPT Image)", "key_hint": "OpenAI API key (platform.openai.com)"},
    "ideogram": {"label": "Ideogram", "key_hint": "Ideogram API key (ideogram.ai/platform/api-keys)"},
    "google": {"label": "Google Gemini (Nano Banana)", "key_hint": "Gemini API key (aistudio.google.com/apikey)"},
    "xai": {"label": "xAI (Grok Imagine)", "key_hint": "xAI API key (console.x.ai)"},
    "fal": {"label": "fal.ai (any model)", "key_hint": "fal API key (fal.ai/dashboard/keys)", "open": True,
            "examples": ["fal-ai/flux-2-pro", "fal-ai/nano-banana-2", "fal-ai/ideogram/v3", "fal-ai/recraft/v3/text-to-image",
                         "fal-ai/qwen-image", "fal-ai/hidream-i1-full"]},
    "replicate": {"label": "Replicate (any model)", "key_hint": "Replicate API token (replicate.com/account/api-tokens)", "open": True,
                  "examples": ["black-forest-labs/flux-2-pro", "google/nano-banana-pro", "ideogram-ai/ideogram-v3-quality",
                               "recraft-ai/recraft-v3", "bytedance/seedream-4"]},
}
for _p, _m in MODELS.items():
    PROVIDERS[_p]["models"] = list(_m)


class ApiError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- http helpers
def _http(method: str, url: str, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None,
          timeout: float = 180) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"User-Agent": "EkphrasisPromptBench/1.0", **headers,
                                          **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        hint = {401: "the API key was refused", 402: "out of credits", 403: "not allowed for this key",
                404: "not found - check the model id", 429: "rate limit - too many requests"}.get(e.code, "")
        raise ApiError(f"HTTP {e.code}{' - ' + hint if hint else ''}: {detail}") from None
    except urllib.error.URLError as e:
        raise ApiError(f"could not reach {url.split('/')[2]}: {e.reason}") from None
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _download(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    req = urllib.request.Request(url, headers={"User-Agent": "EkphrasisPromptBench/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _rval(r: str) -> float:
    a, b = r.split(":")
    return float(a) / float(b)


def nearest_ratio(w: int, h: int, choices: List[str]) -> str:
    real = [c for c in choices if ":" in c]
    if not real:
        return choices[0] if choices else "1:1"
    if not (w and h):
        return "1:1" if "1:1" in real else real[0]
    t = w / h
    return min(real, key=lambda r: abs(math.log(_rval(r) / t)))


def wh_for(ratio: str, megapixels: float, multiple: int = 16, max_side: int = 0) -> Tuple[int, int]:
    r = _rval(ratio) if ":" in ratio else 1.0
    w = math.sqrt(megapixels * 1_000_000 * r)
    h = w / r
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = w * s, h * s
    return max(multiple, int(round(w / multiple)) * multiple), max(multiple, int(round(h / multiple)) * multiple)


def _size(opts: Dict[str, Any], caps: Dict[str, Any]) -> Tuple[str, str]:
    """(aspect_ratio, resolution) for this request: the preset's choice, else nearest to its width x height."""
    ratio = opts.get("aspect_ratio") or ""
    if not ratio or (caps.get("ratios") and ratio not in caps["ratios"]):
        ratio = nearest_ratio(opts.get("width") or 0, opts.get("height") or 0, caps.get("ratios") or ["1:1"])
    res = opts.get("resolution") or ""
    if caps.get("resolutions") and res not in caps["resolutions"]:
        res = caps["resolutions"][min(1, len(caps["resolutions"]) - 1)] if len(caps["resolutions"]) > 1 else caps["resolutions"][0]
    return ratio, res


# ----------------------------------------------------------------------------------------------- fixed providers
def _bfl(prompt, model, seed, opts, caps, key) -> List[bytes]:
    ratio, res = _size(opts, caps)
    body: Dict[str, Any] = {"prompt": prompt}
    if caps["size"] == "ratio_res":                            # FLUX 3: aspect ratio + resolution tier
        body["aspect_ratio"] = ratio
        body["resolution"] = res or "1k"
        if model == "flux-3-image-klein":
            body["output_format"] = "png"
    elif caps["size"] == "ratio":                              # FLUX 1.1 ultra
        body["aspect_ratio"] = ratio if ratio != "auto" else "1:1"
        body["output_format"] = "png"
    else:                                                      # FLUX.2 / 1.1 pro: exact width x height
        mp = float((res or "1MP").replace("MP", ""))
        body["width"], body["height"] = wh_for(ratio if ratio != "auto" else "1:1", mp, caps.get("multiple", 16), caps.get("max_side", 0))
        body["output_format"] = "png"
    if seed is not None and caps.get("seed"):
        body["seed"] = int(seed) % 4294967295
    body.update({k: v for k, v in (opts.get("extra") or {}).items() if k in caps.get("options", {}) or caps.get("loose")})
    sub = _http("POST", f"https://api.bfl.ai/v1/{model}", {"x-key": key, "accept": "application/json"}, body)
    poll = sub.get("polling_url") or f"https://api.bfl.ai/v1/get_result?id={sub.get('id')}"
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(1.5)
        r = _http("GET", poll, {"x-key": key, "accept": "application/json"})
        st = r.get("status") if isinstance(r, dict) else None
        if st == "Ready":
            return [_download(r["result"]["sample"])]
        if st in ("Pending", "Reasoning", "Generating", "Queued", "Processing"):
            continue
        raise ApiError(f"BFL: {st or r} {(r.get('details') if isinstance(r, dict) else '') or ''}".strip())
    raise ApiError("BFL: no result after 10 minutes")


def _openai(prompt, model, seed, opts, caps, key) -> List[bytes]:
    size = opts.get("resolution") if opts.get("resolution") in caps["presets"] else ""
    if not size:
        w, h = opts.get("width") or 0, opts.get("height") or 0
        size = "1536x1024" if w and h and w / h > 1.2 else "1024x1536" if w and h and h / w > 1.2 else "1024x1024"
    body = {"model": model, "prompt": prompt, "size": size, "n": 1}
    if opts.get("quality") and opts["quality"] != "auto":
        body["quality"] = opts["quality"]
    r = _http("POST", "https://api.openai.com/v1/images/generations", {"Authorization": f"Bearer {key}"}, body)
    return [base64.b64decode(d["b64_json"]) if d.get("b64_json") else _download(d["url"]) for d in r.get("data", [])]


def _ideogram(prompt, model, seed, opts, caps, key) -> List[bytes]:
    ratio, res = _size(opts, caps)
    w, h = wh_for(ratio, 1.05 if res != "2K" else 4.19, 32)
    while w * h > 2048 * 2048:
        w, h = w - 32, h - int(32 * h / w)
    body: Dict[str, Any] = {"prompt": prompt, "num_images": 1, "size": f"{w}x{h}"}
    if opts.get("quality"):
        body["quality"] = opts["quality"]
    if seed is not None:
        body["seed"] = int(seed) % 2147483647
    body.update({k: v for k, v in (opts.get("extra") or {}).items() if k in caps.get("options", {})})
    url = f"https://api.ideogram.ai/v2/image/generate/{model}"
    try:
        r = _http("POST", url, {"Api-Key": key}, body)
    except ApiError as e:                                      # not one of Ideogram's preset sizes: let it choose
        if "HTTP 400" in str(e) or "HTTP 422" in str(e):
            body["size"] = "auto"
            r = _http("POST", url, {"Api-Key": key}, body)
        else:
            raise
    out = []
    for d in r.get("data", []):
        if d.get("is_image_safe") is False:
            raise ApiError("Ideogram flagged the image as unsafe and did not return it")
        out.append(_download(d["url"]))
    return out


def _find_images(obj: Any) -> List[str]:
    found: List[str] = []
    if isinstance(obj, dict):
        if obj.get("type") == "image" and isinstance(obj.get("data"), str):
            found.append(obj["data"])
        for v in obj.values():
            found += _find_images(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _find_images(v)
    return found


def _google(prompt, model, seed, opts, caps, key) -> List[bytes]:
    ratio, res = _size(opts, caps)
    fmt: Dict[str, Any] = {"type": "image", "mime_type": "image/png", "aspect_ratio": ratio}
    if res:
        fmt["image_size"] = res
    body = {"model": model, "input": [{"type": "text", "text": prompt}], "response_format": fmt}
    r = _http("POST", "https://generativelanguage.googleapis.com/v1beta/interactions", {"x-goog-api-key": key}, body)
    steps = r.get("steps") if isinstance(r, dict) else None
    imgs = _find_images([s for s in (steps or []) if s.get("type") == "model_output"]) or _find_images(r)
    if not imgs:
        raise ApiError("Gemini returned no image (the prompt may have been declined)")
    return [base64.b64decode(imgs[-1])]


def _xai(prompt, model, seed, opts, caps, key) -> List[bytes]:
    ratio, res = _size(opts, caps)
    body: Dict[str, Any] = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json",
                            "aspect_ratio": ratio, "resolution": res or "1k"}
    if opts.get("quality") and opts["quality"] != "auto":
        body["quality"] = opts["quality"]
    r = _http("POST", "https://api.x.ai/v1/images/generations", {"Authorization": f"Bearer {key}"}, body)
    return [base64.b64decode(d["b64_json"]) if d.get("b64_json") else _download(d["url"]) for d in r.get("data", [])]


# ----------------------------------------------------------------------------------------------- schema readers (fal, Replicate)
def _resolve(node: Any, root: Dict[str, Any]) -> Any:
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 10:
        ref = node["$ref"].lstrip("#/").split("/")
        cur: Any = root
        for part in ref:
            cur = cur.get(part, {}) if isinstance(cur, dict) else {}
        node = {**cur, **{k: v for k, v in node.items() if k != "$ref"}}
        seen += 1
    return node


def _enum(prop: Dict[str, Any], root: Dict[str, Any]) -> List[Any]:
    prop = _resolve(prop, root)
    if "enum" in prop:
        return list(prop["enum"])
    vals: List[Any] = []
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in prop.get(key, []) or []:
            sub = _resolve(sub, root)
            if "enum" in sub:
                vals += sub["enum"]
            elif "const" in sub:
                vals.append(sub["const"])
    return vals


def _accepts_object(prop: Dict[str, Any], root: Dict[str, Any]) -> bool:
    prop = _resolve(prop, root)
    for sub in [prop] + [_resolve(s, root) for s in prop.get("anyOf", []) + prop.get("oneOf", [])]:
        if sub.get("type") == "object" or "width" in (sub.get("properties") or {}):
            return True
    return False


def caps_from_schema(props: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a model's input properties into caps the preset form and the adapter both use."""
    caps = _cap("none", loose=True, fields={})
    if "aspect_ratio" in props:
        caps["ratios"] = [str(v) for v in _enum(props["aspect_ratio"], root)] or R11
        caps["ratio_field"] = "aspect_ratio"
        caps["size"] = "ratio"
    for f in ("resolution", "megapixels", "image_size_tier", "output_resolution"):
        if f in props and _enum(props[f], root):
            caps["resolutions"] = [str(v) for v in _enum(props[f], root)]
            caps["res_field"] = f
            caps["size"] = "ratio_res" if caps["ratios"] else "res"
            break
    if "image_size" in props:
        vals = [str(v) for v in _enum(props["image_size"], root)]
        if vals and not caps["ratios"]:
            caps["presets"] = vals
            caps["preset_field"] = "image_size"
            caps["size"] = "preset"
        if _accepts_object(props["image_size"], root):
            caps["wh_object"] = "image_size"
            if caps["size"] in ("none",):
                caps["size"] = "wh"
    if "width" in props and "height" in props and caps["size"] in ("none", "ratio"):
        caps["wh_fields"] = True
        if caps["size"] == "none":
            caps["size"] = "wh"
    caps["seed"] = "seed" in props
    for f in ("num_images", "num_outputs", "number_of_images"):
        if f in props:
            caps["count_field"] = f
            break
    skip = {"prompt", "seed", "aspect_ratio", "image_size", "width", "height", caps.get("res_field"), caps.get("count_field"),
            "sync_mode", "enable_safety_checker", "webhook"}
    for name, p in props.items():
        if name in skip:
            continue
        p = _resolve(p, root)
        vals = _enum(p, root)
        typ = p.get("type") or ("string" if vals else "")
        if vals:
            caps["fields"][name] = {"type": "enum", "values": [str(v) for v in vals], "default": p.get("default")}
        elif typ in ("integer", "number", "boolean", "string"):
            caps["fields"][name] = {"type": typ, "default": p.get("default"), "min": p.get("minimum"), "max": p.get("maximum")}
    return caps


_SCHEMA_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def fal_caps(model: str) -> Dict[str, Any]:
    hit = _SCHEMA_CACHE.get("fal:" + model)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    spec = _http("GET", "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=" + urllib.parse.quote(model, safe="/"), {}, timeout=40)
    if not isinstance(spec, dict):
        raise ApiError("fal returned no schema for this model id")
    schemas = (spec.get("components") or {}).get("schemas") or {}
    inp = None
    for path, ops in (spec.get("paths") or {}).items():          # the POST that submits a request names the input schema
        body = ((ops.get("post") or {}).get("requestBody") or {}).get("content", {}).get("application/json", {}).get("schema")
        if body:
            inp = _resolve(body, spec)
            break
    if not inp or "properties" not in inp:
        inp = next((s for n, s in schemas.items() if "Input" in n and "prompt" in (s.get("properties") or {})), None)
    if not inp:
        raise ApiError("could not find this fal model's input schema - check the model id (e.g. fal-ai/flux-2-pro)")
    caps = caps_from_schema(inp.get("properties") or {}, spec)
    caps["note"] = (spec.get("info") or {}).get("title", "") or ""
    _SCHEMA_CACHE["fal:" + model] = (time.time(), caps)
    return caps


def replicate_caps(model: str, key: str) -> Dict[str, Any]:
    hit = _SCHEMA_CACHE.get("rep:" + model)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    owner_name = model.split(":")[0]
    if "/" not in owner_name:
        raise ApiError("Replicate model ids look like owner/name (or owner/name:version)")
    if ":" in model:
        m = _http("GET", f"https://api.replicate.com/v1/models/{owner_name}/versions/{model.split(':', 1)[1]}",
                  {"Authorization": f"Bearer {key}"}, timeout=40)
        schema = m.get("openapi_schema") or {}
    else:
        m = _http("GET", f"https://api.replicate.com/v1/models/{owner_name}", {"Authorization": f"Bearer {key}"}, timeout=40)
        schema = ((m.get("latest_version") or {}).get("openapi_schema")) or {}
    inp = ((schema.get("components") or {}).get("schemas") or {}).get("Input") or {}
    if not inp.get("properties"):
        raise ApiError("Replicate returned no input schema for this model")
    caps = caps_from_schema(inp["properties"], schema)
    caps["note"] = (m.get("description") or "")[:200] if isinstance(m, dict) else ""
    _SCHEMA_CACHE["rep:" + model] = (time.time(), caps)
    return caps


def _schema_body(prompt: str, seed, opts: Dict[str, Any], caps: Dict[str, Any]) -> Dict[str, Any]:
    body: Dict[str, Any] = {"prompt": prompt}
    ratio, res = _size(opts, caps)
    if caps.get("ratio_field") and ratio:
        body[caps["ratio_field"]] = ratio
    if caps.get("res_field") and res:
        body[caps["res_field"]] = res
    if caps.get("preset_field"):
        pick = opts.get("resolution") if opts.get("resolution") in caps["presets"] else ""
        if pick:
            body[caps["preset_field"]] = pick
        elif caps.get("wh_object") and opts.get("width") and opts.get("height"):
            body[caps["wh_object"]] = {"width": int(opts["width"]), "height": int(opts["height"])}
    elif caps.get("wh_object") and opts.get("width") and opts.get("height"):
        body[caps["wh_object"]] = {"width": int(opts["width"]), "height": int(opts["height"])}
    if caps.get("wh_fields") and not caps.get("ratio_field") and opts.get("width") and opts.get("height"):
        body["width"], body["height"] = int(opts["width"]), int(opts["height"])
    if caps.get("seed") and seed is not None:
        body["seed"] = int(seed) % 2147483647
    if caps.get("count_field"):
        body[caps["count_field"]] = 1
    body.update(opts.get("extra") or {})
    return body


def _urls(out: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(out, str) and (out.startswith("http") or out.startswith("data:")):
        urls.append(out)
    elif isinstance(out, list):
        for v in out:
            urls += _urls(v)
    elif isinstance(out, dict):
        if isinstance(out.get("url"), str):
            urls.append(out["url"])
        else:
            for k in ("images", "image", "output", "outputs"):
                if k in out:
                    urls += _urls(out[k])
    return urls


def _fal(prompt, model, seed, opts, caps, key) -> List[bytes]:
    body = _schema_body(prompt, seed, opts, caps)
    head = {"Authorization": f"Key {key}"}
    sub = _http("POST", "https://queue.fal.run/" + model, head, body)
    status_url, response_url = sub.get("status_url"), sub.get("response_url")
    if not status_url:
        raise ApiError(f"fal: unexpected reply {str(sub)[:200]}")
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(1.5)
        st = _http("GET", status_url, head)
        if st.get("status") == "COMPLETED":
            if st.get("error"):
                raise ApiError(f"fal: {st.get('error')}")
            out = _http("GET", response_url, head)
            urls = _urls(out)
            if not urls:
                raise ApiError(f"fal returned no image ({str(out)[:200]})")
            if isinstance(out, dict) and any(out.get("has_nsfw_concepts") or []):
                raise ApiError("fal's safety checker blanked the image (has_nsfw_concepts)")
            return [_download(u) for u in urls[:1]]
    raise ApiError("fal: no result after 15 minutes")


def _replicate(prompt, model, seed, opts, caps, key) -> List[bytes]:
    body = {"input": _schema_body(prompt, seed, opts, caps)}
    head = {"Authorization": f"Bearer {key}", "Prefer": "wait=60"}
    if ":" in model:
        p = _http("POST", "https://api.replicate.com/v1/predictions", head, {**body, "version": model})
    else:
        p = _http("POST", f"https://api.replicate.com/v1/models/{model}/predictions", head, body)
    deadline = time.time() + 900
    while p.get("status") not in ("succeeded", "failed", "canceled") and time.time() < deadline:
        time.sleep(2)
        p = _http("GET", p["urls"]["get"], {"Authorization": f"Bearer {key}"})
    if p.get("status") != "succeeded":
        raise ApiError(f"Replicate: {p.get('status')} {p.get('error') or ''}".strip())
    urls = _urls(p.get("output"))
    if not urls:
        raise ApiError("Replicate returned no image")
    auth = {"Authorization": f"Bearer {key}"} if "replicate.delivery" in urls[0] else {}
    return [_download(urls[0], auth)]


ADAPTERS = {"bfl": _bfl, "openai": _openai, "ideogram": _ideogram, "google": _google, "xai": _xai,
            "fal": _fal, "replicate": _replicate}


# ----------------------------------------------------------------------------------------------- public
def model_caps(backend: str, model: str, key: str = "") -> Dict[str, Any]:
    if backend not in PROVIDERS:
        raise ApiError(f"unknown image API: {backend}")
    if backend == "fal":
        return fal_caps(model)
    if backend == "replicate":
        if not key:
            raise ApiError("add your Replicate token in Settings > Image APIs first - it is needed to read the model")
        return replicate_caps(model, key)
    caps = MODELS[backend].get(model)
    if caps is None:                                            # a model id typed in that is not in the table
        return _cap("ratio_res" if backend in ("google", "xai", "ideogram") else "ratio_mp", ratios=R11,
                    resolutions=["1MP", "2MP", "4MP"] if backend == "bfl" else [], loose=True,
                    note="not in the built-in table - using generic settings")
    return caps


def generate(backend: str, prompt: str, seed: Optional[int], model: str, opts: Dict[str, Any], key: str) -> List[Tuple[bytes, str]]:
    if backend not in ADAPTERS:
        raise ApiError(f"unknown image API: {backend}")
    if not key:
        raise ApiError(f"no API key for {PROVIDERS[backend]['label']} - add it in Settings > Image APIs")
    caps = model_caps(backend, model, key)
    datas = ADAPTERS[backend](prompt, model, seed, opts, caps, key)
    if not datas:
        raise ApiError(f"{PROVIDERS[backend]['label']} returned no image")
    return [(d, _ext(d)) for d in datas]
