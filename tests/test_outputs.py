"""Offline tests for naming, grouping, cleaning and no-overwrite. Run: .venv\\Scripts\\python.exe -m tests.test_outputs"""

import os
import tempfile

from vlm_prompter import config
from vlm_prompter.generate import clean_prompt, split_variations
from vlm_prompter.lmstudio import split_think
from vlm_prompter.outputs import group_folder, group_names, shorten_part, unique_path, write_text_new

S = config.load_settings()
GENERIC = {g.lower() for g in S["generic_folders"]}
STOP = {s.lower() for s in S["stop_words"]}


def check(label, got, want):
    status = "ok  " if got == want else "FAIL"
    print(f"{status} {label}: {got!r}" + ("" if got == want else f"  (want {want!r})"))
    return got == want


ok = True
# --- his two examples (SPEC 6a)
root1 = r"D:\Data\styles\graphic-art"
img1 = root1 + r"\1000001-SAMPLE_STYLE_LORA_NAME_LONG_ENOUGH\2000002-V1\images\a.png"
g1 = group_folder(img1, root1, "folder", GENERIC)
ok &= check("generic leaf merges up", g1, root1 + r"\1000001-SAMPLE_STYLE_LORA_NAME_LONG_ENOUGH\2000002-V1")
ok &= check("example 1 name", group_names([g1], root1, S)[g1], "1000001_2000002")

root2 = r"E:\Reference"
img2 = root2 + r"\FashionModelPortfolios\animacreatives\women_angaihi-mumbai\x.jpg"
g2 = group_folder(img2, root2, "folder", GENERIC)
ok &= check("example 2 name", group_names([g2], root2, S)[g2], "FashionModelPortfolios_angaihi-mumbai")

# --- depth grouping
ok &= check("depth:1", group_folder(img2, root2, "depth:1", GENERIC), root2 + r"\FashionModelPortfolios")
ok &= check("depth:1 file in root", group_folder(root2 + r"\y.jpg", root2, "depth:1", GENERIC), root2)

# --- shortening
ok &= check("id wins", shorten_part("2000002-V1", STOP, 24), "2000002")
ok &= check("truncate", shorten_part("A_very_long_folder_name_that_goes_on", STOP, 24), "A_very_long_folder_name")

# --- collisions widen with the middle folder
a = root2 + r"\Portfolios\agencyA\angaihi"
b = root2 + r"\Portfolios\agencyB\angaihi"
n = group_names([a, b], root2, S)
ok &= check("collision widened", (n[a], n[b]), ("Portfolios_agencyA_angaihi", "Portfolios_agencyB_angaihi"))

# --- images directly in root
ok &= check("root itself", group_names([root2], root2, S)[root2], "Reference")
leaf = r"D:\Data\styles\graphic-art\1000001-SAMPLE_STYLE\2000002-V1\images"
ok &= check("start folder is a generic leaf", group_names([leaf], leaf, S)[leaf], "1000001_2000002")

# --- cleaning
ok &= check("label+quotes", clean_prompt('**Prompt:** "A red fox in snow."'), "A red fox in snow.")
ok &= check("bullets", clean_prompt("- a fox\n- in snow"), "a fox\nin snow")
ok &= check("variations", split_variations("### 1\nfirst one\n### 2\nsecond one", 2), ["first one", "second one"])
ok &= check("think split", split_think("<think>hmm</think>  A fox."), ("A fox.", "hmm"))
ok &= check("think close only", split_think("plan...</think>A fox."), ("A fox.", "plan..."))

# --- no-overwrite
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "img.prompt.txt")
    first = write_text_new(p, "a")
    second = write_text_new(p, "b")
    ok &= check("first write", os.path.basename(first), "img.prompt.txt")
    ok &= check("second write", os.path.basename(second), "img.2.prompt.txt")  # keeps *.prompt.txt matchable
    ok &= check("original intact", open(first, encoding="utf-8").read(), "a")

print("\nALL PASS" if ok else "\nSOME FAILED")
