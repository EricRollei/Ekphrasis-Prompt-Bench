"""
LM Studio client: model list, chat completion with stats, explicit load/unload through the lms CLI.

Model list comes from the lms CLI (verified 2026-11-06):
  lms ls --json  -> [{modelKey, type, displayName, publisher, path (relative to the models folder), sizeBytes,
                      quantization{name,bits}, vision (bool), maxContextLength, architecture ...}]
                    `vision` is the reliable flag: the REST API reports DavidAU Fable models as type "llm".
  lms ps --json  -> loaded models: same fields + identifier (the id the API answers to), contextLength, status
The REST API (/api/v0/models) is the fallback when the CLI is unavailable.

Chat: POST /api/v0/chat/completions -> content / reasoning_content, usage (reasoning_tokens, MTP draft counts),
stats (tokens_per_second, time_to_first_token, generation_time, stop_reason), model_info, runtime.

Reasoning control depends on the model's chat template (see model_caps.py):
  reason_tag  (DavidAU TWIN-TURBO): "{REASON:<mode>}" in the user message; i-prefixed modes = no thinking
  effort      (Qwen3.8 style):      reasoning_effort low|medium|xhigh
  thinking    (Qwen3.6 style):      on/off only
  "off" on effort/thinking models is sent as reasoning_effort "none" (tested on Qwen3.6: 0 reasoning tokens).

Standard library only.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LMStudioError(RuntimeError):
    """Raised for connection problems and HTTP errors from LM Studio."""


@dataclass
class ModelInfo:
    id: str                    # modelKey (what you load); for loaded models the API identifier is in `identifier`
    type: str                  # vlm | llm   ("vlm" whenever the model has vision)
    state: str                 # loaded | not-loaded
    quantization: str = ""
    max_context: int = 0
    arch: str = ""
    publisher: str = ""
    display: str = ""
    path: str = ""             # absolute path of the GGUF when known
    size_bytes: int = 0
    vision: bool = False
    identifier: str = ""       # loaded models: the id to send to the chat API
    loaded_context: int = 0

    @property
    def loaded(self) -> bool:
        return self.state == "loaded"


@dataclass
class ChatResult:
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    generation_time: float = 0.0
    wall_time: float = 0.0
    stop_reason: str = ""
    model_info: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    draft: Dict[str, int] = field(default_factory=dict)

    @property
    def starved(self) -> bool:
        """Ran out of tokens before producing any visible answer (typically thinking ate the budget)."""
        return not self.content.strip() and self.finish_reason == "length"


def thinks(mode: str) -> bool:
    """Does this reasoning mode produce thinking tokens? (off and i-prefixed instruct modes do not)"""
    return bool(mode) and mode != "off" and not mode.startswith("i")


class LMStudioClient:
    def __init__(self, host: str = "http://localhost:1234", lms_exe: str = "", timeout_s: float = 900,
                 models_dir: str = ""):
        self.host = host.rstrip("/")
        self.lms_exe = lms_exe
        self.timeout_s = timeout_s
        self.models_dir = models_dir

    # ------------------------------------------------------------------------------------------------ HTTP helpers
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        url = self.host + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LMStudioError(f"LM Studio HTTP {exc.code} on {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LMStudioError(f"LM Studio not reachable at {self.host} ({exc.reason}). Is the server running?") from exc

    # ------------------------------------------------------------------------------------------------ lms CLI
    def _lms(self, args: List[str], timeout: float = 600) -> subprocess.CompletedProcess:
        if not self.lms_exe or not os.path.isfile(self.lms_exe):
            raise LMStudioError(f"lms CLI not found at {self.lms_exe!r} (set lms_exe in settings.json)")
        return subprocess.run([self.lms_exe] + args, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _lms_json(self, args: List[str]) -> List[Dict[str, Any]]:
        proc = self._lms(args, timeout=60)
        text = (proc.stdout or "").strip()
        start = text.find("[")
        if proc.returncode != 0 or start < 0:
            raise LMStudioError(f"lms {' '.join(args)} failed: {(proc.stderr or text)[-300:]}")
        return json.loads(text[start:])

    def _abs_path(self, rel: str) -> str:
        if not rel:
            return ""
        if os.path.isabs(rel) or not self.models_dir:
            return rel
        return os.path.join(self.models_dir, rel.replace("/", os.sep))

    # ------------------------------------------------------------------------------------------------ models
    def loaded_models(self) -> List[ModelInfo]:
        """Models loaded right now (lms ps), API fallback."""
        try:
            rows = self._lms_json(["ps", "--json"])
            return [self._from_cli(r, loaded=True) for r in rows if r.get("type") != "embedding"]
        except (LMStudioError, ValueError, OSError, subprocess.SubprocessError):
            return [m for m in self._api_models(include_all=True) if m.loaded]

    def list_models(self, include_all: bool = False) -> List[ModelInfo]:
        """Vision models by default (include_all adds text-only ones). Loaded state merged in."""
        try:
            rows = self._lms_json(["ls", "--json"])
        except (LMStudioError, ValueError, OSError, subprocess.SubprocessError):
            return self._api_models(include_all)
        loaded = {m.id: m for m in self.loaded_models()}
        out = []
        for r in rows:
            if r.get("type") in ("embedding", "embeddings"):
                continue
            m = self._from_cli(r, loaded=False)
            if m.id in loaded:
                m.state, m.identifier, m.loaded_context = "loaded", loaded[m.id].identifier, loaded[m.id].loaded_context
            if include_all or m.vision:
                out.append(m)
        return out

    def _from_cli(self, r: Dict[str, Any], loaded: bool) -> ModelInfo:
        q = r.get("quantization") or {}
        vision = bool(r.get("vision"))
        return ModelInfo(
            id=r.get("modelKey") or r.get("identifier") or "", type="vlm" if vision else (r.get("type") or "llm"),
            state="loaded" if loaded else "not-loaded", quantization=q.get("name", "") if isinstance(q, dict) else str(q),
            max_context=int(r.get("maxContextLength") or 0), arch=r.get("architecture") or "",
            publisher=r.get("publisher") or "", display=r.get("displayName") or "", path=self._abs_path(r.get("path") or ""),
            size_bytes=int(r.get("sizeBytes") or 0), vision=vision,
            identifier=(r.get("identifier") or r.get("modelKey") or "") if loaded else "",
            loaded_context=int(r.get("contextLength") or 0) if loaded else 0,
        )

    def _api_models(self, include_all: bool) -> List[ModelInfo]:
        raw = self._request("GET", "/api/v0/models", timeout=15).get("data", [])
        out = []
        for m in raw:
            mtype = m.get("type", "")
            if mtype == "embeddings" or (not include_all and mtype != "vlm"):
                continue
            out.append(ModelInfo(
                id=m.get("id", ""), type=mtype, state=m.get("state", ""), quantization=m.get("quantization", "") or "",
                max_context=int(m.get("max_context_length") or 0), arch=m.get("arch", "") or "",
                publisher=m.get("publisher", "") or "", vision=mtype == "vlm",
                identifier=m.get("id", "") if m.get("state") == "loaded" else "",
                loaded_context=int(m.get("loaded_context_length") or 0)))
        return out

    def loaded_model_ids(self) -> List[str]:
        ids = []
        for m in self.loaded_models():
            ids += [m.id, m.identifier]
        return [i for i in ids if i]

    # ------------------------------------------------------------------------------------------------ chat
    def chat(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.7,
             top_p: Optional[float] = None, seed: Optional[int] = None, reasoning: str = "off",
             caps: Optional[Dict[str, bool]] = None, sampling: Optional[Dict[str, float]] = None) -> ChatResult:
        """
        reasoning: a mode from model_caps.modes_for() ("off", "on", low/medium/xhigh, einstein/spoon, i-prefixed).
        caps: {"reason_tag", "effort", "thinking"} from the model's chat template.
        sampling: optional top_k, min_p, presence_penalty, repeat_penalty.
        """
        caps = caps or {}
        messages = copy.deepcopy(messages)
        body: Dict[str, Any] = {"model": model, "max_tokens": int(max_tokens), "temperature": float(temperature),
                                "stream": False}
        if top_p is not None:
            body["top_p"] = float(top_p)
        if seed is not None:
            body["seed"] = int(seed)
        for k in ("top_k", "min_p", "presence_penalty", "repeat_penalty"):
            v = (sampling or {}).get(k)
            if v is not None:
                body[k] = int(v) if k == "top_k" else float(v)

        mode = reasoning or "off"
        if caps.get("reason_tag"):
            tag_mode = "imedium" if mode == "off" else ("xhigh" if mode == "on" else mode)
            _prepend_user_text(messages, "{REASON:" + tag_mode + "} ")
        elif not thinks(mode):
            body["reasoning_effort"] = "none"
        elif caps.get("effort") and mode in ("low", "medium", "xhigh"):
            body["reasoning_effort"] = mode
        body["messages"] = messages

        start = time.perf_counter()
        resp = self._request("POST", "/api/v0/chat/completions", body)
        wall = time.perf_counter() - start

        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = resp.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        stats = resp.get("stats") or {}
        content, inline_reasoning = split_think(msg.get("content") or "")
        reasoning_text = (msg.get("reasoning_content") or "") + inline_reasoning
        return ChatResult(
            content=content, reasoning=reasoning_text, finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            tokens_per_second=float(stats.get("tokens_per_second") or 0.0),
            time_to_first_token=float(stats.get("time_to_first_token") or 0.0),
            generation_time=float(stats.get("generation_time") or 0.0), wall_time=wall,
            stop_reason=stats.get("stop_reason") or "", model_info=resp.get("model_info") or {},
            runtime=resp.get("runtime") or {},
            draft={k: int(usage[k]) for k in ("total_draft_tokens_count", "accepted_draft_tokens_count",
                                               "rejected_draft_tokens_count") if k in usage},
        )

    # ------------------------------------------------------------------------------------------------ load / unload
    def load(self, model: str, context_length: Optional[int] = None) -> float:
        """Load a model through `lms load`; returns load time in seconds (0.0 if it was already loaded)."""
        if model in self.loaded_model_ids():
            return 0.0
        args = ["load", model, "-y"]
        if context_length:
            args += ["-c", str(int(context_length))]
        start = time.perf_counter()
        proc = self._lms(args)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0 or model not in self.loaded_model_ids():
            tail = (proc.stdout + proc.stderr).strip()[-600:]
            raise LMStudioError(f"lms load {model} failed (code {proc.returncode}): {tail}")
        return elapsed

    def unload(self, model: str) -> None:
        self._lms(["unload", model], timeout=120)

    def unload_all(self) -> None:
        self._lms(["unload", "--all"], timeout=120)

    def ensure_loaded(self, model: str, context_length: Optional[int] = None, exclusive: bool = True) -> float:
        """Load `model` if needed (unloading the others first when exclusive). Returns load seconds."""
        loaded = self.loaded_models()
        if any(model in (m.id, m.identifier) for m in loaded):
            return 0.0
        if exclusive:
            for other in loaded:
                self.unload(other.identifier or other.id)
        return self.load(model, context_length)


def _prepend_user_text(messages: List[Dict[str, Any]], text: str) -> None:
    """Put `text` at the start of the last user message (string or multi-part content)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = text + content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    part["text"] = text + part.get("text", "")
                    break
            else:
                content.insert(0, {"type": "text", "text": text})
        return


def split_think(text: str) -> "tuple[str, str]":
    """Separate inline <think>...</think> blocks (some chat templates leave them in content)."""
    if "<think>" not in text and "</think>" not in text:
        return text, ""
    reasoning_parts = []
    out = text
    while "</think>" in out:
        head, _, tail = out.partition("</think>")
        if "<think>" in head:
            before, _, inner = head.partition("<think>")
        else:                                   # template opened the block in the prompt; only the close is here
            before, inner = "", head
        reasoning_parts.append(inner)
        out = before + tail
    if "<think>" in out:                        # unterminated block = starved mid-thought
        out, _, inner = out.partition("<think>")
        reasoning_parts.append(inner)
    return out.strip(), "\n".join(p.strip() for p in reasoning_parts)
