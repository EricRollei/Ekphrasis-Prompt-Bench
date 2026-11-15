"""
Refresh the vendored wildcard engine (vlm_prompter/wildcards) from video_prompter's core/wildcards.

    python tools/sync_wildcards.py [path to video_prompter]          # shows what would change
    python tools/sync_wildcards.py [path to video_prompter] --apply  # copies the changed files

The path can also come from the VIDEO_PROMPTER_DIR environment variable. Only the engine's .py files are copied;
Ekphrasis' adapter (vlm_prompter/wc.py) is not touched.

(c) Eric Hiss. Dual license, see LICENSE.md.
"""

import filecmp
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(os.path.dirname(HERE), "vlm_prompter", "wildcards")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    root = args[0] if args else os.environ.get("VIDEO_PROMPTER_DIR", "")
    source = os.path.join(root, "core", "wildcards")
    if not os.path.isdir(source):
        print(f"not found: {source}\nusage: python tools/sync_wildcards.py <video_prompter folder> [--apply]")
        return 1
    changed = []
    for name in sorted(os.listdir(source)):
        if not name.endswith(".py"):
            continue
        src, dst = os.path.join(source, name), os.path.join(TARGET, name)
        if not os.path.exists(dst):
            changed.append((name, "new"))
        elif not filecmp.cmp(src, dst, shallow=False):
            changed.append((name, "changed"))
    if not changed:
        print("the vendored engine is up to date")
        return 0
    for name, what in changed:
        print(f"  {what:8} {name}")
        if apply:
            shutil.copy2(os.path.join(source, name), os.path.join(TARGET, name))
    print("copied." if apply else "run again with --apply to copy these files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
