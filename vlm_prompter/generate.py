"""
One image -> prompt(s).

Flow: hash image -> cache lookup -> prepare image -> chat -> clean -> (starved? one retry with thinking off and a
doubled budget) -> split variations -> cache + log.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .images import file_hash, prepare_image
from .instructions import Composed, Preset
from .lmstudio import ChatResult, LMStudioClient, LMStudioError, thinks
from .store import ResultCache, RunLog, cache_key


@dataclass
class GenParams:
    model: str
    max_tokens: int = 0              # 0 = preset's max_tokens
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    reasoning: str = ""              # "" = preset's
    max_side: int = 1536
    snap: int = 32
    jpeg_quality: int = 92
    reasoning_budget: int = 4096
    caps: Dict[str, bool] = field(default_factory=dict)     # model's reasoning controls (model_caps.caps_for)
    layout: Dict[str, Any] = field(default_factory=dict)    # canvas for layout presets: {"aspect", "mp"}


@dataclass
class PromptResult:
    image: str
    prompts: List[str] = field(default_factory=list)
    cached: bool = False
    starved: bool = False
    retried: bool = False
    over_limit: bool = False
    error: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.prompts) and not self.error


_LABEL_RE = re.compile(r"^\s*(\*\*)?\s*(final\s+)?(prompt|krea\s*2?\s*prompt|image\s+prompt)\s*(\d+)?\s*:?\s*(\*\*)?\s*:?\s*",
                       re.IGNORECASE)
_VARIANT_RE = re.compile(r"^\s*#{2,4}\s*(\d+)\s*[.:)]?\s*$", re.MULTILINE)


def clean_prompt(text: str) -> str:
    """Strip labels, wrapping quotes, markdown and bullets the model added despite instructions."""
    t = text.strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t)
    t = _LABEL_RE.sub("", t, count=1)
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"^[-*\u2022]\s+", "", s)
        s = s.replace("**", "").replace("__", "")
        lines.append(s)
    t = "\n".join(lines).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) > 1 and t[0] in "\"'\u201c" and t[-1] in "\"'\u201d":
        t = t[1:-1].strip()
    return t


def split_variations(text: str, expected: int) -> List[str]:
    if expected <= 1:
        cleaned = clean_prompt(text)
        return [cleaned] if cleaned else []
    parts = _VARIANT_RE.split(text)
    # split gives [preamble, n1, body1, n2, body2, ...]
    bodies = [clean_prompt(parts[i + 1]) for i in range(1, len(parts) - 1, 2)] if len(parts) > 2 else []
    bodies = [b for b in bodies if b]
    if not bodies:                    # model ignored the numbering; fall back to blank-line blocks
        bodies = [clean_prompt(b) for b in re.split(r"\n\s*\n", text) if clean_prompt(b)]
    return bodies


def extract_json(text: str) -> str:
    """The JSON object/array inside a model answer (fences and chatter stripped), re-serialised compactly.
    Raises ValueError when there is no parseable JSON."""
    t = re.sub(r"```[a-zA-Z]*", "", text).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object in the answer")
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    if end <= start:
        raise ValueError("JSON is cut off (no closing bracket) - raise max tokens")
    obj = json.loads(t[start:end + 1])
    return json.dumps(obj, ensure_ascii=False)


def _messages(composed: Composed, data_uri: str, note: str = "") -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": composed.system},
        {"role": "user", "content": [
            {"type": "text", "text": composed.user + ("\n\n" + note if note else "")},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]},
    ]


def _stats(r: ChatResult) -> Dict[str, Any]:
    return {
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "reasoning_tokens": r.reasoning_tokens, "output_tokens": r.completion_tokens - r.reasoning_tokens,
        "tok_s": round(r.tokens_per_second, 2), "ttft_s": round(r.time_to_first_token, 3),
        "gen_s": round(r.generation_time, 2), "wall_s": round(r.wall_time, 2),
        "finish": r.finish_reason, "stop_reason": r.stop_reason, "draft": r.draft,
        "quant": r.model_info.get("quant", ""), "arch": r.model_info.get("arch", ""),
        "context_length": r.model_info.get("context_length", 0), "runtime": r.runtime.get("version", ""),
    }


def generate_one(client: LMStudioClient, image_path: str, preset: Preset, composed: Composed, params: GenParams,
                 cache: Optional[ResultCache] = None, log: Optional[RunLog] = None,
                 force: bool = False) -> PromptResult:
    result = PromptResult(image=image_path)
    reasoning = params.reasoning or preset.reasoning
    temperature = preset.temperature if params.temperature is None else params.temperature
    top_p = preset.top_p if params.top_p is None else params.top_p
    sampling = {"top_k": preset.top_k, "min_p": preset.min_p, "presence_penalty": preset.presence_penalty,
                "repeat_penalty": preset.repeat_penalty}
    max_tokens = (params.max_tokens or preset.max_tokens) * max(1, composed.variations)
    if thinks(reasoning):
        max_tokens += params.reasoning_budget

    try:
        img_hash = file_hash(image_path)
    except OSError as exc:
        result.error = f"cannot read image: {exc}"
        return result

    key = cache_key(image=img_hash, model=params.model, instr=composed.sha, max_side=params.max_side,
                    snap=params.snap, temperature=temperature, top_p=top_p, seed=params.seed,
                    reasoning=reasoning, max_tokens=max_tokens, variations=composed.variations, vision="single",
                    sampling=sampling)
    record_base = {"image": image_path, "image_hash": img_hash, "model": params.model, "preset": preset.id,
                   "instr_sha": composed.sha, "parts": composed.parts, "reasoning": reasoning,
                   "temperature": temperature, "top_p": top_p, "seed": params.seed, "max_tokens": max_tokens,
                   "variations": composed.variations, "cache_key": key}

    if cache and not force:
        hit = cache.get(key)
        if hit and hit.get("prompts"):
            result.prompts, result.stats, result.cached = hit["prompts"], hit.get("stats", {}), True
            result.over_limit = bool(hit.get("over_limit"))
            if log:
                log.write({**record_base, "cached": True, "prompts": result.prompts})
            return result

    try:
        prepared = prepare_image(image_path, params.max_side, params.snap, params.jpeg_quality)
    except Exception as exc:  # PIL raises many types for broken files
        result.error = f"cannot open image: {exc}"
        if log:
            log.write({**record_base, "error": result.error})
        return result

    from .images import is_video, video_note
    messages = _messages(composed, prepared.data_uri, video_note(image_path) if is_video(image_path) else "")
    try:
        chat = client.chat(params.model, messages, max_tokens=max_tokens, temperature=temperature,
                           top_p=top_p, seed=params.seed, reasoning=reasoning, caps=params.caps, sampling=sampling)
        if chat.starved:
            result.starved = True
            result.retried = True
            chat = client.chat(params.model, messages, max_tokens=max_tokens * 2, temperature=temperature,
                               top_p=top_p, seed=params.seed, reasoning="off", caps=params.caps, sampling=sampling)
        elif chat.finish_reason == "length" and (chat.content or "").strip():
            # cut off mid-prompt (Nov 14 to-do): one retry with a bigger budget; keep the longer answer
            retry = client.chat(params.model, messages, max_tokens=int(max_tokens * 1.8), temperature=temperature,
                                top_p=top_p, seed=params.seed, reasoning=reasoning, caps=params.caps, sampling=sampling)
            result.retried = True
            if retry.finish_reason != "length" or len(retry.content or "") > len(chat.content or ""):
                chat = retry
    except LMStudioError as exc:
        result.error = str(exc)
        if log:
            log.write({**record_base, "error": result.error})
        return result

    if preset.output == "json":
        pieces = (split_variations(chat.content, composed.variations) if composed.variations > 1
                  else [chat.content])
        try:
            result.prompts = [extract_json(p) for p in pieces if p.strip()]
            if preset.layout:                     # boxes 0-1000 -> the target model's canvas coordinates
                from .layout import apply as apply_layout
                conv = [apply_layout(preset.layout, p, params.layout, prepared.original_size) for p in result.prompts]
                result.prompts = [c[0] for c in conv]
                result.stats["canvas"] = conv[0][1] if conv else ""
        except ValueError as exc:          # json.JSONDecodeError is a ValueError too
            result.prompts = []
            result.error = f"invalid JSON: {exc}"
    else:
        result.prompts = split_variations(chat.content, composed.variations)
    canvas = result.stats.get("canvas")
    result.stats = _stats(chat)
    if canvas:
        result.stats["canvas"] = canvas
    result.stats.update({"sent_size": prepared.sent_size, "original_size": prepared.original_size})
    if not result.prompts and not result.error:
        result.error = f"empty answer (finish={chat.finish_reason})"
    if preset.token_limit:
        per_prompt = result.stats["output_tokens"] / max(1, len(result.prompts) or 1)
        result.over_limit = per_prompt > preset.token_limit

    if log:
        log.write({**record_base, "cached": False, "prompts": result.prompts, "stats": result.stats,
                   "starved_first_try": result.starved, "over_limit": result.over_limit,
                   "raw_content": chat.content if not result.prompts else None,
                   "reasoning_chars": len(chat.reasoning), "error": result.error or None,
                   "sent_size": prepared.sent_size})
    if cache and result.ok:
        cache.put(key, {"prompts": result.prompts, "stats": result.stats, "over_limit": result.over_limit,
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"), **record_base})
    return result


def runaway(text: str) -> str:
    """'' if the text looks sane, else why it looks like a runaway repetition loop (Nov 15): a very long stretch of
    words with no punctuation, or a long phrase repeated word for word."""
    import re as _re
    words = (text or "").split()
    if len(words) < 120:
        return ""
    run = best = 0
    for w in words:
        if _re.search(r"[.,;:!?)\]\"]$", w):
            run = 0
        else:
            run += 1
            best = max(best, run)
    if best >= 70:
        return f"runaway text ({best} words without punctuation)"
    low = [w.lower().strip(".,;:!?\"'") for w in words]
    seen = set()
    for i in range(len(low) - 9):
        g = tuple(low[i:i + 10])
        if g in seen:
            return "runaway text (a long phrase repeats word for word)"
        seen.add(g)
    return ""
