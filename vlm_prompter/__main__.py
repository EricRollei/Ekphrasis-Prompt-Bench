"""
VLM Prompter CLI.

    python -m vlm_prompter serve [--port 8790] [--no-browser]       the GUI
    python -m vlm_prompter models [--all]
    python -m vlm_prompter presets
    python -m vlm_prompter run <paths...> [--model M] [--preset krea2] [--mod id[=value]]... [--note TEXT]
                               [--out per-image,per-folder,combined] [--format lines|blocks] [--group-by folder|depth:N]
                               [--group-dir DIR] [--combined-path FILE] [--variations N] [--reasoning off|on|auto]
                               [--max-tokens N] [--temperature T] [--seed S] [--no-recursive] [--skip-existing]
                               [--force] [--dry-run] [--no-load]
    python -m vlm_prompter combine <paths...> [--out FILE] [--pattern *.prompt.txt] [--format lines|blocks] [--dedupe]

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from typing import Dict, List, Tuple

from . import config
from .images import discover, natural_key
from .instructions import load_modifiers, load_presets
from .jobs import OUT_MODES, JobSpec, make_client, plan_outputs, run_generation, write_outputs
from .lmstudio import LMStudioError
from .outputs import format_entries, one_line, write_text_new


def cmd_models(args, settings) -> int:
    from .model_caps import caps_for, modes_for
    favs = set(settings.get("favorite_models", []))
    models = make_client(settings).list_models(include_all=args.all)
    models.sort(key=lambda m: (m.id not in favs, not m.loaded, m.id.lower()))
    for m in models:
        flags = ("*" if m.id in favs else " ") + ("L" if m.loaded else " ")
        size = f"{m.size_bytes / 1e9:5.1f} GB" if m.size_bytes else ""
        line = f"{flags} {m.id:<90} {m.quantization:<7} {size}"
        if args.modes:
            line += "  modes: " + ",".join(v for v, _ in modes_for(caps_for(m.path)))
        print(line)
    print(f"\n{len(models)} models  (* favourite, L loaded)")
    return 0


def cmd_presets(args, settings) -> int:
    for p in load_presets(config.PRESETS_DIR).values():
        print(f"preset   {p.id:<20} {p.name}  [tokens {p.token_target[0]}-{p.token_target[1]}, reasoning {p.reasoning}]")
    for m in load_modifiers(config.MODIFIERS_DIR).values():
        print(f"modifier {m.id:<20} {m.text}")
    return 0


def _pick_model(args, settings) -> str:
    if args.model:
        return args.model
    return "__loaded__"                                   # default: whatever LM Studio has loaded


def _print_event(e: Dict) -> None:
    t = e["type"]
    if t == "load" and e["state"] == "ready":
        print(f"model {e['model']}: " + (f"loaded in {e['load_s']}s" if e.get("load_s") else "already loaded"))
    elif t == "item":
        label = f"[{e['i']}/{e['total']}] {e['image']}"
        if e["status"] == "skipped":
            print(f"{label}  skipped (prompt file exists)")
        elif e["status"] == "error":
            print(f"{label}  ERROR {e['error']}")
        elif e["status"] == "cached":
            print(f"{label}  cached")
        else:
            s = e["stats"]
            flags = (" STARVED-RETRIED" if e.get("retried") else "") + (" OVER-LIMIT" if e.get("over_limit") else "")
            print(f"{label}  {s.get('output_tokens', 0)} tok  {s.get('wall_s', 0)}s  {s.get('tok_s', 0)} tok/s{flags}")


def cmd_run(args, settings) -> int:
    modes = [m.strip() for m in args.out.split(",") if m.strip()]
    bad = [m for m in modes if m not in OUT_MODES]
    if bad:
        raise SystemExit(f"unknown --out mode(s): {bad}; choose from {OUT_MODES}")
    items = discover(args.paths, recursive=not args.no_recursive)
    if not items:
        print("No images found.")
        return 1

    group_dir = args.group_dir or settings["group_dir"]
    plan = plan_outputs(settings, items, args.group_by)
    print(f"{len(items)} images in {len(plan['groups'])} group(s)")
    if args.dry_run or "per-folder" in modes:
        for g in plan["groups"]:
            print(f"  {g['count']:>5}  {g['name']:<50} <- {g['group']}")
    if args.dry_run:
        if "combined" in modes:
            print(f"  combined -> {os.path.join(group_dir, plan['combined'])}")
        return 0

    model = _pick_model(args, settings)
    spec = JobSpec(items=items, model=model, preset=args.preset or "", mods=args.mod, note=args.note or "",
                   variations=args.variations, reasoning=args.reasoning or "", max_tokens=args.max_tokens or 0,
                   temperature=args.temperature, seed=args.seed, force=args.force, skip_existing=args.skip_existing,
                   load=not args.no_load, live_per_image="per-image" in modes, facts=args.facts,
                   facts_ethnicity=args.facts_ethnicity, argv=sys.argv[1:])
    settings["last_model"] = model
    config.save_settings(settings)
    try:
        summary = run_generation(settings, spec, _print_event)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    prompts = {img: r.prompts for img, r in summary["results"].items() if r.ok}
    meta = {img: {"model": model, "preset": summary["preset"], "instr": summary["instr_sha"]} for img in prompts}
    written = list(summary["written"])
    written += write_outputs(settings, items, prompts, [m for m in modes if m != "per-image"], args.format,
                             args.group_by, group_dir, args.combined_path or "", None,
                             "append" if args.append else "new", args.dedupe, meta, args.sidecar)
    summary["log_obj"].write({"event": "end", "files": written, "errors": summary["errors"],
                              "elapsed_s": summary["elapsed_s"]})
    print(f"\n{len(prompts)} prompted, {summary['errors']} errors, {summary['elapsed_s']:.0f}s. Log: {summary['log']}")
    per_suffixes = (settings["per_image_suffix"], settings["per_image_suffix_json"])
    for w in written:
        if not w["path"].lower().endswith(per_suffixes):
            print(f"  {w['action']:<9} {w['path']}  ({w['count']} prompts)")
    n_img = sum(1 for w in written if w["path"].lower().endswith(per_suffixes))
    if n_img:
        print(f"  wrote {n_img} per-image prompt file(s)")
    return 0 if not summary["errors"] else 3


def cmd_combine(args, settings) -> int:
    files: List[str] = []
    for raw in args.paths:
        p = os.path.abspath(raw)
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames.sort(key=natural_key)
                files += [os.path.join(dirpath, f) for f in sorted(filenames, key=natural_key)
                          if fnmatch.fnmatch(f.lower(), args.pattern.lower())]
    if not files:
        print("No prompt files found.")
        return 1
    seen = set()
    entries: List[Tuple[str, List[str]]] = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as handle:
            lines = [one_line(l) for l in handle.read().splitlines() if l.strip() and not l.startswith("#")]
        if args.dedupe:
            lines = [l for l in lines if not (l.lower() in seen or seen.add(l.lower()))]
        if lines:
            entries.append((f, lines))
    try:
        base = os.path.commonpath([os.path.dirname(f) for f in files])
    except ValueError:                                 # different drives
        base = os.path.dirname(files[0])
    sep = settings["name_separator"]
    out = args.out or os.path.join(settings["group_dir"], f"{os.path.basename(base) or 'prompts'}{sep}combined{sep}prompts.txt")
    path = write_text_new(out, format_entries(entries, args.format, base))
    print(f"{sum(len(e[1]) for e in entries)} prompts from {len(entries)} files -> {path}")
    return 0


def cmd_whence(args, settings) -> int:
    from .store import search_index
    text = args.text if args.text != "-" else sys.stdin.read()
    hits = search_index(config.RUNS_DIR, text)
    if not hits:
        print("No saved prompt matches (only prompts saved since the index started are known).")
        return 1
    for h in hits:
        print(f"[{h['match']}] {h.get('saved', '')}  {h.get('model', '?')}  preset {h.get('preset', '?')}  instructions {h.get('instr', '?')}")
        print(f"    image: {h.get('image', '')}")
        print(f"    file:  {h.get('file', '')}")
        ins = h.get("instructions") or {}
        if ins.get("modifiers") or ins.get("note"):
            print(f"    modifiers: {ins.get('modifiers')}  note: {ins.get('note')!r}")
    return 0


def cmd_serve(args, settings) -> int:
    from .server import serve
    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vlm_prompter", description="Image -> text-to-image prompts via LM Studio VLMs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="start the GUI")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--no-browser", action="store_true")

    p = sub.add_parser("models", help="list LM Studio models")
    p.add_argument("--all", action="store_true", help="include text-only models")
    p.add_argument("--modes", action="store_true", help="also show each model's reasoning modes (reads templates)")

    sub.add_parser("presets", help="list presets and modifiers")

    p = sub.add_parser("whence", help="which model / instructions / image produced a saved prompt")
    p.add_argument("text", help="the prompt text (or a long fragment), or - to read stdin")

    p = sub.add_parser("run", help="generate prompts for images / folders")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model")
    p.add_argument("--preset")
    p.add_argument("--mod", action="append", default=[], help="modifier id or id=value (repeatable)")
    p.add_argument("--note", help="extra instruction for every image in this run")
    p.add_argument("--out", default="per-image", help="comma list of per-image, per-folder, combined")
    p.add_argument("--combined-path", help="explicit path for the combined file")
    p.add_argument("--group-dir", help="folder for per-folder/combined files (default: settings group_dir)")
    p.add_argument("--format", choices=("lines", "blocks", "jsonl"), default="lines")
    p.add_argument("--append", action="store_true", help="add to existing list files instead of making new numbered ones")
    p.add_argument("--dedupe", action="store_true", help="skip prompts already in the target list file")
    p.add_argument("--sidecar", action="store_true", help="also write name.meta.jsonl beside each list (model etc. per line)")
    p.add_argument("--group-by", default="folder", help="folder | depth:N")
    p.add_argument("--variations", type=int, default=1)
    p.add_argument("--facts", action="store_true", help="send measured facts (pose + face detectors) with each image")
    p.add_argument("--facts-ethnicity", action="store_true", help="include the detector's ethnicity guess in the facts")
    p.add_argument("--reasoning", help="off, on, low/medium/xhigh, einstein/spoon, or i-prefixed (ixhigh ...); see models --modes")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="skip images that already have a .prompt.txt")
    p.add_argument("--force", action="store_true", help="ignore the cache")
    p.add_argument("--dry-run", action="store_true", help="show groups and file names only")
    p.add_argument("--no-load", action="store_true", help="don't load/unload models; let LM Studio JIT-load")

    p = sub.add_parser("combine", help="merge single prompt files into one file")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out")
    p.add_argument("--pattern", default="*.prompt.txt")
    p.add_argument("--format", choices=("lines", "blocks", "jsonl"), default="lines")
    p.add_argument("--dedupe", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = config.load_settings()
    handler = {"serve": cmd_serve, "models": cmd_models, "presets": cmd_presets, "run": cmd_run,
               "combine": cmd_combine, "whence": cmd_whence}[args.cmd]
    try:
        return handler(args, settings)
    except (KeyError, ValueError, LMStudioError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
