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
            if is_media(path) and path not in seen:
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
                    if is_media(full) and full not in seen:
                        seen.add(full)
                        root_items.append(full)
        else:
            for name in sorted(os.listdir(path), key=natural_key):
                full = os.path.join(path, name)
                if os.path.isfile(full) and is_media(full) and full not in seen:
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
    if is_video(path):                               # a clip reaches the vision model as a contact sheet of frames
        return contact_sheet(path)
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


# ----------------------------------------------------------------------------------------------- video (Nov 23)
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi"}
SHEET_FRAMES = 6


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_media(path: str) -> bool:
    return is_image(path) or is_video(path)


def video_frames(path: str, n: int = SHEET_FRAMES) -> List[Tuple[float, Image.Image]]:
    """n frames spread across the clip, as (seconds, RGB image) - OpenCV, no external tools."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        picks = [int(total * (k + 0.5) / n) for k in range(n)] if total > n else [0]
        out: List[Tuple[float, Image.Image]] = []
        for idx in picks:
            if total > n:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out.append((idx / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        if not out:
            raise ValueError(f"no frames could be read from {os.path.basename(path)}")
        return out
    finally:
        cap.release()


def video_duration(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        return total / fps if total > 0 else 0.0
    finally:
        cap.release()


def video_poster(path: str) -> Image.Image:
    """A representative frame (about a tenth of the way in) for thumbnails."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 10)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"cannot read a frame from {os.path.basename(path)}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def contact_sheet(path: str, n: int = SHEET_FRAMES, cols: int = 3, cell: int = 640) -> Image.Image:
    """The clip as one picture: n frames in time order (left to right, top to bottom), numbered with their time."""
    from PIL import ImageDraw
    frames = video_frames(path, n)
    w0, h0 = frames[0][1].size
    cw = cell
    ch = max(1, int(cell * h0 / w0))
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw + (cols - 1) * 6, rows * ch + (rows - 1) * 6), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, (t, im) in enumerate(frames):
        x, y = (k % cols) * (cw + 6), (k // cols) * (ch + 6)
        sheet.paste(im.resize((cw, ch), Image.LANCZOS), (x, y))
        label = f"{k + 1}  {t:.1f}s"
        draw.rectangle([x, y, x + 9 * len(label) + 10, y + 22], fill=(0, 0, 0))
        draw.text((x + 6, y + 5), label, fill=(255, 255, 255))
    return sheet


def video_note(path: str) -> str:
    try:
        dur = video_duration(path)
    except Exception:
        dur = 0.0
    length = f"{dur:.0f}-second " if dur else ""
    return (f"Note: the picture is a contact sheet of {SHEET_FRAMES} frames sampled from a {length}video clip, in time "
            "order (left to right, top to bottom, numbered with their time). Treat it as the video itself: describe the "
            "scene and what moves and changes over time, not a grid of separate pictures.")
