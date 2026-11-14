"""
Image discovery and preparation for the VLM.

prepare_image: EXIF rotation, first frame of animations, alpha flattened onto mid grey, long side capped at max_side,
both sides snapped DOWN to a multiple of `snap` (min 64) - the qwen3vl-captioner clip.cpp "Invalid number of output
tokens" crash was image-dimension dependent and went away with 32-snapping. Encoded as JPEG data URI.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif"}

try:                                            # optional formats if the plugins are installed
    import pillow_avif  # noqa: F401
    IMAGE_EXTS.add(".avif")
except ImportError:
    pass
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    IMAGE_EXTS.update({".heic", ".heif"})
except ImportError:
    pass


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def discover(paths: Iterable[str], recursive: bool = True) -> List[Tuple[str, str]]:
    """
    Expand files/folders into (image_path, root) pairs, sorted naturally within each root.
    A folder argument is its own root; a single file's root is its parent folder.
    Folders whose name starts with '.' or '_' are skipped (caches, our own _prompts, quarantine folders).
    """
    found: List[Tuple[str, str]] = []
    seen = set()
    for raw in paths:
        path = os.path.abspath(raw)
        if os.path.isfile(path):
            if is_image(path) and path not in seen:
                seen.add(path)
                found.append((path, os.path.dirname(path)))
            continue
        if not os.path.isdir(path):
            print(f"WARNING: not found: {raw}")
            continue
        root_items = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = sorted((d for d in dirnames if not d.startswith((".", "_"))), key=natural_key)
                for name in sorted(filenames, key=natural_key):
                    full = os.path.join(dirpath, name)
                    if is_image(full) and full not in seen:
                        seen.add(full)
                        root_items.append(full)
        else:
            for name in sorted(os.listdir(path), key=natural_key):
                full = os.path.join(path, name)
                if os.path.isfile(full) and is_image(full) and full not in seen:
                    seen.add(full)
                    root_items.append(full)
        found.extend((p, path) for p in root_items)
    return found


def natural_key(text: str):
    """img2 before img10."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PreparedImage:
    data_uri: str
    sent_size: Tuple[int, int]
    original_size: Tuple[int, int]


def snap_size(width: int, height: int, max_side: int, snap: int) -> Tuple[int, int]:
    scale = min(1.0, float(max_side) / max(width, height))
    w, h = width * scale, height * scale
    w = max(64, int(w // snap) * snap)
    h = max(64, int(h // snap) * snap)
    return w, h


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (128, 128, 128))
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


def prepare_image(path: str, max_side: int = 1536, snap: int = 32, quality: int = 92) -> PreparedImage:
    img = load_rgb(path)
    original = img.size
    target = snap_size(original[0], original[1], max_side, snap)
    if target != original:
        img = img.resize(target, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return PreparedImage(data_uri=uri, sent_size=target, original_size=original)
