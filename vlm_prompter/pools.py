"""
Saved Photobash pools: a pool library you can save, search and load again.

Each saved pool is a folder  <app>/pools/<id>/  holding COPIES of its hand-picked images (resized to at most 2048 px
on the long side, high-quality JPEG) and a pool.json:
  {id, name, description, created, updated,
   folders: [folder paths the pool draws from - kept as references],
   images:  [{file (copy, relative), original, folder, name}]}
Copies keep a pool working when the originals are moved; whole folders stay references (they can hold hundreds of
images) and are reported as missing when a load cannot find them. The pools folder is private (git-ignored).

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import uuid
from typing import Any, Dict, List, Optional

from . import config

POOLS_DIR = os.path.join(config.APP_DIR, "pools")
MAX_SIDE = 2048


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip()).strip("-.")
    return (s or "pool")[:48]


def _manifest(pid: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(POOLS_DIR, pid, "pool.json")
    try:
        with open(path, "r", encoding="utf-8") as h:
            return json.load(h)
    except (OSError, ValueError):
        return None


def _write_manifest(pid: str, m: Dict[str, Any]) -> None:
    path = os.path.join(POOLS_DIR, pid, "pool.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as h:
        json.dump(m, h, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _copy_image(src: str, dst_dir: str, n: int) -> str:
    """Copy src into dst_dir as a <=2048 px JPEG; return the new file name."""
    from PIL import Image, ImageOps
    base = os.path.splitext(os.path.basename(src))[0]
    name = f"{n:03d}_{_slug(base)[:60]}.jpg"
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(os.path.join(dst_dir, name), "JPEG", quality=92, optimize=True)
    return name


def save_pool(name: str, description: str, files: List[str], folders: List[str],
              pool_id: Optional[str] = None) -> Dict[str, Any]:
    """Save (or, with pool_id, update) a pool. files may mix originals and copies already in this pool."""
    name = (name or "").strip()
    if not name:
        raise ValueError("give the pool a name")
    files = [f for f in files if f]
    folders = [f for f in folders if f]
    if not files and not folders:
        raise ValueError("the pool is empty")
    os.makedirs(POOLS_DIR, exist_ok=True)
    old = _manifest(pool_id) if pool_id else None
    if pool_id and not old:
        pool_id = None
    if not pool_id:
        pool_id = f"{_dt.date.today().strftime('%Y%m%d')}_{_slug(name)}_{uuid.uuid4().hex[:6]}"
    pdir = os.path.join(POOLS_DIR, pool_id)
    os.makedirs(pdir, exist_ok=True)
    known = {os.path.normcase(os.path.join(pdir, im["file"])): im for im in (old or {}).get("images", [])}
    images, used, skipped = [], set(), []
    n = max([int(im["file"][:3]) for im in known.values() if im["file"][:3].isdigit()] + [0])
    for f in files:
        key = os.path.normcase(os.path.abspath(f))
        if key in known:                                   # a copy already in this pool: keep it
            images.append(known[key])
            used.add(key)
            continue
        if not os.path.isfile(f):
            skipped.append(f)
            continue
        n += 1
        try:
            fname = _copy_image(f, pdir, n)
        except Exception:
            skipped.append(f)
            continue
        used.add(os.path.normcase(os.path.join(pdir, fname)))
        images.append({"file": fname, "original": f, "folder": os.path.dirname(f), "name": os.path.basename(f)})
    for key, im in known.items():                          # copies taken out of the pool
        if key not in used:
            try:
                os.remove(os.path.join(pdir, im["file"]))
            except OSError:
                pass
    m = {"id": pool_id, "name": name, "description": (description or "").strip(),
         "created": (old or {}).get("created") or _now(), "updated": _now(), "folders": folders, "images": images}
    _write_manifest(pool_id, m)
    return {**_card(m), "skipped": skipped}


def _card(m: Dict[str, Any]) -> Dict[str, Any]:
    pdir = os.path.join(POOLS_DIR, m["id"])
    imgs = m.get("images", [])
    src_folders = sorted({os.path.basename(im.get("folder", "")) for im in imgs if im.get("folder")})
    return {"id": m["id"], "name": m.get("name", ""), "description": m.get("description", ""),
            "created": m.get("created", ""), "updated": m.get("updated", ""), "count": len(imgs),
            "folders": m.get("folders", []), "source_folders": src_folders,
            "thumbs": [os.path.join(pdir, im["file"]) for im in imgs[:4]]}


def list_pools(query: str = "") -> List[Dict[str, Any]]:
    if not os.path.isdir(POOLS_DIR):
        return []
    words = [w for w in (query or "").lower().split() if w]
    out = []
    for pid in os.listdir(POOLS_DIR):
        m = _manifest(pid)
        if not m:
            continue
        hay = " ".join([m.get("name", ""), m.get("description", ""), " ".join(m.get("folders", [])),
                        " ".join(im.get("original", "") for im in m.get("images", [])),
                        m.get("created", "")]).lower()
        if all(w in hay for w in words):
            out.append(_card(m))
    out.sort(key=lambda c: c["updated"], reverse=True)
    return out


def load_pool(pool_id: str) -> Dict[str, Any]:
    m = _manifest(pool_id)
    if not m:
        raise ValueError("pool not found")
    pdir = os.path.join(POOLS_DIR, pool_id)
    files = [os.path.join(pdir, im["file"]) for im in m.get("images", []) if os.path.isfile(os.path.join(pdir, im["file"]))]
    folders = [f for f in m.get("folders", []) if os.path.isdir(f)]
    missing = [f for f in m.get("folders", []) if not os.path.isdir(f)]
    return {**_card(m), "files": files, "folders": folders, "missing_folders": missing}


def delete_pool(pool_id: str) -> Dict[str, Any]:
    if not _manifest(pool_id):
        raise ValueError("pool not found")
    shutil.rmtree(os.path.join(POOLS_DIR, pool_id))
    return {"ok": True}
