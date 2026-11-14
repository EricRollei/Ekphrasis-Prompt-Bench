"""
Layout (bounding-box) prompts: canvas size from an aspect ratio + megapixels, and conversion of the model's boxes into
the target model's coordinate system.

The vision model writes boxes as [y_min, x_min, y_max, x_max] on a 0-1000 grid of the whole picture (what Qwen-family
models ground reliably). MOSS-Image layout JSON wants absolute PIXELS with the canvas CENTRE as origin
("bbox_coord_space": "abs_centralized", "bbox_format": "yxyx" - OpenMOSS USERGUIDE.md), so every layout prompt
belongs to one canvas size: render at that size.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

ASPECTS = {"1:1": 1.0, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9, "21:9": 21 / 9, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16}


def canvas_size(layout: Optional[Dict[str, Any]], image_wh: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(W, H) in pixels, multiples of 16. layout = {"aspect": "source" | "4:3" ..., "mp": 1.0}."""
    layout = layout or {}
    aspect = str(layout.get("aspect") or "source")
    try:
        mp = float(layout.get("mp") or 1.0)
    except (TypeError, ValueError):
        mp = 1.0
    mp = min(4.0, max(0.25, mp))                      # MOSS: 0.25 MP (512x512) to 4 MP
    if aspect in ASPECTS:
        ar = ASPECTS[aspect]
    elif image_wh and image_wh[0] and image_wh[1]:
        ar = image_wh[0] / image_wh[1]
    else:
        ar = 1.0
    w = math.sqrt(mp * 1_000_000 * ar)
    h = w / ar
    return max(16, int(round(w / 16)) * 16), max(16, int(round(h / 16)) * 16)


def to_moss(text: str, w: int, h: int) -> str:
    """Model JSON with 0-1000 yxyx boxes -> MOSS layout JSON with absolute centred pixel boxes for a WxH canvas."""
    data = json.loads(text)
    items = data.get("items") or []
    for it in items:
        b = it.get("bbox")
        if not (isinstance(b, list) and len(b) == 4):
            it.pop("bbox", None)
            continue
        try:
            y0, x0, y1, x1 = [max(0.0, min(1000.0, float(v))) for v in b]
        except (TypeError, ValueError):
            it.pop("bbox", None)
            continue
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 < x0:
            x0, x1 = x1, x0
        it["bbox"] = [round(y0 / 1000 * h - h / 2), round(x0 / 1000 * w - w / 2),
                      round(y1 / 1000 * h - h / 2), round(x1 / 1000 * w - w / 2)]
    out = {"summary": data.get("summary") or {}, "items": items,
           "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def apply(layout_kind: str, text: str, layout: Optional[Dict[str, Any]],
          image_wh: Optional[Tuple[int, int]] = None) -> Tuple[str, str]:
    """Post-process a layout preset's answer -> (text, "WxH")."""
    if layout_kind != "moss":
        return text, ""
    w, h = canvas_size(layout, image_wh)
    return to_moss(text, w, h), f"{w}x{h}"
