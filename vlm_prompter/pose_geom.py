"""
Pose geometry on COCO-WholeBody keypoints (133 points, as produced by RTMW / DWPose):
facing direction of a person, and similarity of two poses including the mirror-image test.

Keypoint indices used (COCO body part of the 133):
  0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear, 5 left shoulder, 6 right shoulder, 7 left elbow,
  8 right elbow, 9 left wrist, 10 right wrist, 11 left hip, 12 right hip, 13 left knee, 14 right knee,
  15 left ankle, 16 right ankle;  23-90 face landmarks.
"Left" / "right" are the PERSON's own sides, as the detector labels them. Image x grows to the right.

Pure numpy - no detector needed here, so it is unit-testable on its own.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

MIN_SCORE = 0.35
BODY = list(range(17))
# swap table for the 17 body points (person's left <-> right)
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _ok(scores: np.ndarray, *idx: int) -> bool:
    return all(scores[i] >= MIN_SCORE for i in idx)


def facing(kps: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """
    Facing of one person. Returns:
      body: "toward camera" | "away from camera" | "profile" | "unknown"
      turn: "frontal" | "three-quarter" | "profile" | "unknown"     (how far the shoulders are turned)
      side: "left" | "right" | ""     (the side of the PICTURE the person / face is turned toward)
      head: "toward camera" | "away from camera" | "turned left" | "turned right" | "unknown"
      text: one plain sentence for prompts / reports
    """
    out: Dict[str, object] = {"body": "unknown", "turn": "unknown", "side": "", "head": "unknown"}
    # --- shoulder order: the person's left shoulder appears on the picture's RIGHT when they face the camera
    sh = _ok(scores, 5, 6)
    hp = _ok(scores, 11, 12)
    torso = None
    if sh and hp:
        sm = (kps[5] + kps[6]) / 2
        hm = (kps[11] + kps[12]) / 2
        torso = float(np.linalg.norm(sm - hm)) or None
    order = []
    if sh:
        order.append(np.sign(kps[5][0] - kps[6][0]))
    if hp:
        order.append(np.sign(kps[11][0] - kps[12][0]))
    width_ratio = None
    if sh and torso:
        width_ratio = abs(kps[5][0] - kps[6][0]) / torso
        out["turn"] = "frontal" if width_ratio > 0.6 else "three-quarter" if width_ratio > 0.3 else "profile"
    face_pts = int(np.sum(scores[23:91] >= MIN_SCORE)) if len(scores) > 90 else 0
    face_seen = _ok(scores, 0) and (_ok(scores, 1) or _ok(scores, 2))
    if order and out["turn"] != "profile":
        s = np.mean(order)
        if s > 0:
            out["body"] = "toward camera"
        elif s < 0:
            out["body"] = "away from camera"
    elif out["turn"] == "profile":
        out["body"] = "profile"
    # a visible face contradicts "away" only weakly (head can turn); no face + ears = away
    if out["body"] == "unknown" and not face_seen and (_ok(scores, 3) or _ok(scores, 4)) and face_pts < 5:
        out["body"] = "away from camera"
    # --- head: visible face -> toward / turned; nose offset vs ear (or eye) midpoint gives the side
    if face_seen or face_pts >= 20:
        ref = None
        if _ok(scores, 3, 4):
            ref = (kps[3] + kps[4]) / 2
            span = abs(kps[3][0] - kps[4][0])
        elif _ok(scores, 1, 2):
            ref = (kps[1] + kps[2]) / 2
            span = abs(kps[1][0] - kps[2][0]) * 2.2
        if ref is not None and _ok(scores, 0):
            off = (kps[0][0] - ref[0]) / max(span, 1.0)
            if off < -0.18:
                out["head"], out["side"] = "turned left", "left"
            elif off > 0.18:
                out["head"], out["side"] = "turned right", "right"
            else:
                out["head"] = "toward camera"
        elif _ok(scores, 0) and (_ok(scores, 3) != _ok(scores, 4)):        # one ear only: strong profile
            ear = kps[3] if _ok(scores, 3) else kps[4]
            out["head"] = "turned left" if kps[0][0] < ear[0] else "turned right"
            out["side"] = "left" if kps[0][0] < ear[0] else "right"
        else:
            out["head"] = "toward camera"
    elif _ok(scores, 3) or _ok(scores, 4):
        out["head"] = "away from camera"
    if not out["side"] and out["turn"] in ("three-quarter", "profile") and _ok(scores, 0) and sh:
        mid = (kps[5][0] + kps[6][0]) / 2
        sw = max(abs(kps[5][0] - kps[6][0]), 1.0)
        if abs(kps[0][0] - mid) / sw > 0.15:                 # only with a clear nose offset
            out["side"] = "left" if kps[0][0] < mid else "right"
    # --- sentence
    parts = []
    if out["body"] != "unknown":
        b = out["body"]
        if out["turn"] == "three-quarter" and out["side"]:
            b += f", turned three-quarter toward the {out['side']} of the frame"
        elif out["body"] == "profile" and out["side"]:
            b = f"in profile, facing the {out['side']} of the frame"
        parts.append("body " + b)
    if out["head"] != "unknown" and not (out["body"] == "profile" and out["head"].startswith("turned")):
        h = out["head"]
        if h.startswith("turned"):
            h = f"turned to the {out['side']} of the frame"
        parts.append("head " + h)
    out["text"] = "; ".join(parts) or "facing could not be determined"
    return out


def _norm_pose(kps: np.ndarray, scores: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Body points centred on the torso and scaled by torso length; None when the torso is not visible."""
    if not (_ok(scores, 5, 6, 11, 12)):
        return None
    center = (kps[5] + kps[6] + kps[11] + kps[12]) / 4
    torso = np.linalg.norm((kps[5] + kps[6]) / 2 - (kps[11] + kps[12]) / 2)
    if torso < 1:
        return None
    pts = (kps[:17] - center) / torso
    vis = scores[:17] >= MIN_SCORE
    return pts, vis


def compare_poses(src: Tuple[np.ndarray, np.ndarray], gen: Tuple[np.ndarray, np.ndarray]) -> Dict[str, object]:
    """
    Similarity of the source pose and the render pose (main person each), plus the mirror test:
    flip the render horizontally (and swap left/right labels) - if that matches the source clearly better,
    the render shows the pose mirror-imaged.
    Returns {similarity 0-100, mirrored bool, direct, mirror (mean normalised distances), joints}.
    """
    a = _norm_pose(*src)
    b = _norm_pose(*gen)
    if a is None or b is None:
        return {"similarity": None, "mirrored": False, "reason": "torso not visible in one of the images"}
    pa, va = a
    pb, vb = b
    both = va & vb
    if both.sum() < 6:
        return {"similarity": None, "mirrored": False, "reason": "too few joints visible in both"}
    direct = float(np.mean(np.linalg.norm(pa[both] - pb[both], axis=1)))
    pm = pb[SWAP].copy()
    pm[:, 0] *= -1
    vm = vb[SWAP]
    both_m = va & vm
    mirror = float(np.mean(np.linalg.norm(pa[both_m] - pm[both_m], axis=1))) if both_m.sum() >= 6 else 9.0
    mirrored = mirror < direct * 0.6 and direct > 0.2       # clearly better when flipped, and not near-identical
    best = min(direct, mirror) if mirrored else direct
    return {"similarity": round(max(0.0, 100.0 * (1.0 - best / 1.0)), 1), "mirrored": bool(mirrored),
            "direct": round(direct, 3), "mirror": round(mirror, 3), "joints": int(both.sum())}


def main_person(people_kps: np.ndarray, people_scores: np.ndarray) -> Optional[int]:
    """Index of the most prominent person: largest visible-keypoint bounding box weighted by confidence."""
    best, best_i = 0.0, None
    for i, (k, s) in enumerate(zip(people_kps, people_scores)):
        vis = s[:17] >= MIN_SCORE
        if vis.sum() < 4:
            continue
        pts = k[:17][vis]
        area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])) * float(np.mean(s[:17][vis]))
        if area > best:
            best, best_i = area, i
    return best_i


# ===================================================================================================== several people
def person_box(kps: np.ndarray, scores: np.ndarray) -> Optional[List[float]]:
    """Box around the visible body keypoints (x1, y1, x2, y2); None when too few points."""
    vis = scores[:17] >= MIN_SCORE
    if vis.sum() < 3:
        return None
    pts = kps[:17][vis]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    y1 -= 0.12 * (y2 - y1)                      # keypoints stop at the eyes / ears - add the top of the head
    return [float(x1), float(y1), float(x2), float(y2)]


def horiz_dir(f: Dict[str, object]) -> str:
    """Where a person is turned, for relations: 'left' / 'right' (side of the picture), 'viewer', 'away', ''.
    Head turn wins over body (where someone looks carries the meaning in a conversation)."""
    if not f:
        return ""
    if str(f.get("head", "")).startswith("turned") and f.get("side"):
        return str(f["side"])
    if (f.get("body") == "profile" or f.get("turn") == "three-quarter") and f.get("side"):
        return str(f["side"])
    if f.get("body") == "toward camera" or f.get("head") == "toward camera":
        return "viewer"
    if f.get("body") == "away from camera":
        return "away"
    return ""


RELATION_TEXT = {
    "each_other": "face each other",
    "away_from_each_other": "are turned away from each other",
    "a_toward_b": "the left one is turned toward the right one",
    "b_toward_a": "the right one is turned toward the left one",
    "both_viewer": "both face the viewer",
    "both_away": "both have their backs to the viewer",
}


def relation(a_dir: str, b_dir: str) -> str:
    """Relation class for two people, a standing LEFT of b in the picture."""
    if a_dir == "right" and b_dir == "left":
        return "each_other"
    if a_dir == "left" and b_dir == "right":
        return "away_from_each_other"
    if a_dir == "right":
        return "a_toward_b"
    if b_dir == "left":
        return "b_toward_a"
    if a_dir == b_dir == "viewer":
        return "both_viewer"
    if a_dir == b_dir == "away":
        return "both_away"
    return ""


# ===================================================================================================== tolerance
# Directions are coarse buckets; a person turned ~45 degrees sits on a border where either neighbouring answer is fair
# (Eric, Nov 13). Only OPPOSITE directions are a real contradiction.
_OPPOSITE = {frozenset(("left", "right")), frozenset(("toward", "away")), frozenset(("viewer", "away"))}
_TOWARD = {"each_other", "a_toward_b", "b_toward_a"}


def directions_agree(a: str, b: str) -> bool:
    """'toward' vs 'left' agree (neighbours); 'left' vs 'right' and 'toward' vs 'away' do not."""
    a = "toward" if a == "viewer" else a
    b = "toward" if b == "viewer" else b
    return frozenset((a, b)) not in _OPPOSITE


def relations_agree(a: str, b: str) -> bool:
    """Contradiction only when one says someone is turned toward the other and the other says they are turned away
    from each other, or both-facing-the-viewer vs both-backs-to-the-viewer."""
    if not a or not b or a == b:
        return True
    if (a in _TOWARD and b == "away_from_each_other") or (b in _TOWARD and a == "away_from_each_other"):
        return False
    return frozenset((a, b)) != frozenset(("both_viewer", "both_away"))
