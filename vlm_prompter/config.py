"""
Settings for VLM Prompter.

Defaults live in DEFAULTS below; your own values (folders, favourites ...) are saved to settings.json in the app
folder, which is not part of the repository.
Only keys that differ from the defaults need to be in settings.json - missing keys fall back to DEFAULTS.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
PRESETS_DIR = os.path.join(APP_DIR, "presets")
MODIFIERS_DIR = os.path.join(APP_DIR, "modifiers")
RUNS_DIR = os.path.join(APP_DIR, "runs")
CACHE_DIR = os.path.join(APP_DIR, ".cache")

DEFAULTS: Dict[str, Any] = {
    # --- LM Studio -------------------------------------------------------------------------------------------------
    "lmstudio_host": "http://localhost:1234",
    "lms_exe": os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", "lms.exe"),
    "lmstudio_models_dir": os.path.join(os.path.expanduser("~"), ".lmstudio", "models"),   # LM Studio's models folder
    "request_timeout_s": 900,
    "load_context_length": 21000,       # context used when this app loads a model (images + tiles need room)
    "exclusive_load": True,             # unload other loaded models before loading the chosen one (VRAM)
    "reasoning_budget": 4096,           # extra max_tokens added when reasoning is not "off" (thinking eats tokens)

    # --- Generation defaults (a preset can override; the CLI/GUI can override the preset) ---------------------------
    "default_preset": "krea2",
    "last_model": "",
    "favorite_models": [],

    # --- Images ----------------------------------------------------------------------------------------------------
    "max_side": 1536,                   # long side sent to the VLM
    "snap": 32,                         # both sides snapped to a multiple of this (clip.cpp crash lesson)
    "jpeg_quality": 92,

    # --- Outputs ---------------------------------------------------------------------------------------------------
    "group_dir": os.path.join(APP_DIR, "prompt_lists"),   # prompt lists (per folder, combined, photobash ...)
    "wildcards_dir": os.path.join(APP_DIR, "wildcards"),   # Ideas tab: folder of __name__ wildcard .txt files
    "per_image_suffix": ".prompt.txt",               # <stem>.prompt.txt beside the image; never <stem>.txt
    "per_image_suffix_json": ".prompt.json",         # used instead when the preset outputs JSON
    "name_separator": "_",
    "name_part_max": 24,
    "generic_folders": [
        "images", "image", "imgs", "img", "pics", "pictures", "photos", "photo", "samples", "sample",
        "previews", "preview", "renders", "render", "output", "outputs", "gallery", "examples", "example",
        "workflows", "workflow", "embedded_media",
    ],
    "stop_words": [
        "women", "woman", "men", "man", "girls", "girl", "boys", "boy", "female", "male", "model", "models",
    ],
    "group_name_overrides": {},         # {group folder path: name} - set from the GUI preview table

    # --- Prompt Review -----------------------------------------------------------------------------------------------
    "sweeps_dir": "",                   # Review: Eric_Krea2 Sweep Plan output folder (ComfyUI/output/sweeps)
    "review_folders": [],               # extra image folders added in the Review tab
    "recent_folders": [],
}


def load_settings() -> Dict[str, Any]:
    """DEFAULTS merged with settings.json (settings.json wins). A broken settings.json is reported, not fatal."""
    settings = copy.deepcopy(DEFAULTS)
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
                user = json.load(handle)
            if isinstance(user, dict):
                settings.update(user)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: settings.json could not be read ({exc}); using defaults")
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Write only the keys that differ from DEFAULTS, so new defaults reach existing installs."""
    changed = {k: v for k, v in settings.items() if DEFAULTS.get(k, object()) != v}
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(changed, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, SETTINGS_PATH)
