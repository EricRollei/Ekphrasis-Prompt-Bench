"""
Hosted image-generation backends for rendering (Phase 2 of private_docs/SPEC_render.md).

One small adapter per provider: generate(prompt, width, height, seed, model, quality, key) -> list of (bytes, ext).
Each maps the requested width x height onto what that provider accepts. API keys live only in the local
settings.json (api_keys), which is never committed.

Request shapes checked against each provider's official docs on 2026-11-21:
  Black Forest Labs  POST https://api.bfl.ai/v1/{model}  (x-key)  -> poll polling_url -> result.sample (URL, 10 min)
  OpenAI             POST https://api.openai.com/v1/images/generations (Bearer) -> data[].b64_json
  Ideogram           POST https://api.ideogram.ai/v2/image/generate/{model} (Api-Key) -> data[].url (24 h)
  Google Gemini      POST https://generativelanguage.googleapis.com/v1beta/interactions (x-goog-api-key)
                     -> steps[].content[] {type: image, data: base64}
  xAI                POST https://api.x.ai/v1/images/generations (Bearer) -> data[].b64_json / url

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "bfl": {"label": "Black Forest Labs (FLUX)", "key_hint": "BFL API key (dashboard.bfl.ai)",
            "models": ["flux-2-pro-preview", "flux-2-pro", "flux-2-max", "flux-2-flex", "flux-2-klein-9b",
                       "flux-pro-1.1-ultra", "flux-pro-1.1"],
            "qualities": [], "max_n": 1},
    "openai": {"label": "OpenAI (GPT Image)", "key_hint": "OpenAI API key (platform.openai.com)",
               "models": ["gpt-image-2.6-flare", "gpt-image-2.6-sunburst"],
               "qualities": ["auto", "low", "medium", "high"], "max_n": 10},
    "ideogram": {"label": "Ideogram", "key_hint": "Ideogram API key (ideogram.ai/platform/api-keys)",
                 "models": ["ideogram-4-5", "ideogram-4-6", "ideogram-4", "ideogram-3"],
                 "qualities": ["high", "medium", "low"], "max_n": 8},
    "google": {"label": "Google Gemini (Nano Banana)", "key_hint": "Gemini API key (aistudio.google.com/apikey)",
               "models": ["gemini-nano-banana-2.1", "gemini-3-pro-image", "gemini-3.1-flash-image",
                          "gemini-3.1-flash-lite-image"],
               "qualities": ["1K", "2K", "4K"], "max_n": 1},
    "xai": {"label": "xAI (Grok Imagine)", "key_hint": "xAI API key (console.x.ai)",
            "models": ["grok-imagine-image-2.0"],
            "qualities": ["auto", "low", "medium"], "max_n": 10},
}


class ApiError(RuntimeError):
    pass


def _http(method: str, url: str, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None,
          timeout: float = 180) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={**headers, **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        hint = {401: "the API key was refused", 402: "out of credits", 403: "not allowed for this key",
                429: "rate limit - too many requests"}.get(e.code, "")
        raise ApiError(f"HTTP {e.code}{' - ' + hint if hint else ''}: {detail}") from None
    except urllib.error.URLError as e:
        raise ApiError(f"could not reach {url.split('/')[2]}: {e.reason}") from None
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _download(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r:
        return r.read()


def _ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _ratio(w: int, h: int, choices: List[str]) -> str:
    """Nearest aspect ratio string ('16:9') to w x h."""
    if not (w and h):
        return "1:1"
    target = w / h

    def val(r: str) -> float:
        a, b = r.split(":")
        return float(a) / float(b)
    return min(choices, key=lambda r: abs(val(r) - target))


# ----------------------------------------------------------------------------------------------- adapters
def _bfl(prompt, w, h, seed, model, quality, key) -> List[bytes]:
    w, h = (w or 1024), (h or 1024)
    w, h = max(256, w - w % 16), max(256, h - h % 16)        # FLUX wants multiples of 16
    body: Dict[str, Any] = {"prompt": prompt, "width": w, "height": h, "output_format": "png"}
    if seed is not None:
        body["seed"] = int(seed) % 2 ** 32
    if model == "flux-pro-1.1-ultra":                          # ultra takes an aspect ratio, not a size
        body.pop("width"); body.pop("height")
        body["aspect_ratio"] = _ratio(w, h, ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"])
    sub = _http("POST", f"https://api.bfl.ai/v1/{model}", {"x-key": key, "accept": "application/json"}, body)
    poll = sub.get("polling_url") or f"https://api.bfl.ai/v1/get_result?id={sub.get('id')}"
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(1.0)
        r = _http("GET", poll, {"x-key": key, "accept": "application/json"})
        st = r.get("status")
        if st == "Ready":
            return [_download(r["result"]["sample"])]
        if st in ("Error", "Failed", "Content Moderated", "Request Moderated"):
            raise ApiError(f"BFL: {st} {r.get('details') or ''}".strip())
    raise ApiError("BFL: no result after 5 minutes")


def _openai(prompt, w, h, seed, model, quality, key) -> List[bytes]:
    size = "1024x1024"
    if w and h:
        size = "1536x1024" if w / h > 1.2 else "1024x1536" if h / w > 1.2 else "1024x1024"
    body = {"model": model, "prompt": prompt, "size": size, "n": 1}
    if quality and quality != "auto":
        body["quality"] = quality
    r = _http("POST", "https://api.openai.com/v1/images/generations", {"Authorization": f"Bearer {key}"}, body)
    out = []
    for d in r.get("data", []):
        out.append(base64.b64decode(d["b64_json"]) if d.get("b64_json") else _download(d["url"]))
    return out


def _ideogram(prompt, w, h, seed, model, quality, key) -> List[bytes]:
    body: Dict[str, Any] = {"prompt": prompt, "num_images": 1}
    if quality:
        body["quality"] = quality
    if seed is not None:
        body["seed"] = int(seed) % 2147483647
    url = f"https://api.ideogram.ai/v2/image/generate/{model}"
    exact = (w and h and w % 32 == 0 and h % 32 == 0 and w * h <= 2048 * 2048)
    if exact:
        body["size"] = f"{w}x{h}"
    try:
        r = _http("POST", url, {"Api-Key": key}, body)
    except ApiError as e:                                      # an exact size that is not a supported preset
        if exact and ("HTTP 400" in str(e) or "HTTP 422" in str(e)):
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


def _google(prompt, w, h, seed, model, quality, key) -> List[bytes]:
    fmt: Dict[str, Any] = {"type": "image", "mime_type": "image/png",
                           "aspect_ratio": _ratio(w, h, ["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16",
                                                         "16:9", "21:9"])}
    if quality in ("1K", "2K", "4K") and "lite" not in model:
        fmt["image_size"] = quality
    body = {"model": model, "input": [{"type": "text", "text": prompt}], "response_format": fmt}
    r = _http("POST", "https://generativelanguage.googleapis.com/v1beta/interactions", {"x-goog-api-key": key}, body)
    steps = r.get("steps") if isinstance(r, dict) else None
    imgs = _find_images([s for s in (steps or []) if s.get("type") == "model_output"]) or _find_images(r)
    if not imgs:
        raise ApiError("Gemini returned no image (the prompt may have been declined)")
    return [base64.b64decode(imgs[-1])]                        # the last image is the final one


def _xai(prompt, w, h, seed, model, quality, key) -> List[bytes]:
    body: Dict[str, Any] = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json",
                            "aspect_ratio": _ratio(w, h, ["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]),
                            "resolution": "2k" if max(w or 0, h or 0) > 1400 else "1k"}
    if quality and quality != "auto":
        body["quality"] = quality
    r = _http("POST", "https://api.x.ai/v1/images/generations", {"Authorization": f"Bearer {key}"}, body)
    out = []
    for d in r.get("data", []):
        out.append(base64.b64decode(d["b64_json"]) if d.get("b64_json") else _download(d["url"]))
    return out


ADAPTERS = {"bfl": _bfl, "openai": _openai, "ideogram": _ideogram, "google": _google, "xai": _xai}


def generate(backend: str, prompt: str, width: int, height: int, seed: Optional[int], model: str,
             quality: str, key: str) -> List[Tuple[bytes, str]]:
    if backend not in ADAPTERS:
        raise ApiError(f"unknown image API: {backend}")
    if not key:
        raise ApiError(f"no API key for {PROVIDERS[backend]['label']} - add it in Settings > Image APIs")
    datas = ADAPTERS[backend](prompt, int(width or 0), int(height or 0), seed, model, quality, key)
    if not datas:
        raise ApiError(f"{PROVIDERS[backend]['label']} returned no image")
    return [(d, _ext(d)) for d in datas]


def ideogram_quote(prompt: str, width: int, height: int, model: str, quality: str, key: str, n: int) -> Optional[float]:
    """Ideogram's dry_run price quote for n images (None if it cannot be had)."""
    try:
        body: Dict[str, Any] = {"prompt": prompt or "a test", "num_images": 1}
        if quality:
            body["quality"] = quality
        r = _http("POST", f"https://api.ideogram.ai/v2/image/generate/{model}?dry_run=true", {"Api-Key": key}, body, 30)
        for k in ("price_usd", "total_usd", "cost_usd", "price", "total"):
            if isinstance(r, dict) and isinstance(r.get(k), (int, float)):
                return float(r[k]) * n
    except Exception:
        return None
    return None
