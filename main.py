#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "dghs-imgutils",
#   "onnxruntime",
#   "insightface",
#   "opencv-python>=4.9",
#   "numpy>=2.0",
# ]
# ///
"""
Anime eye censor bar with rotation.

Face detection : dghs-imgutils  (anime-specific ONNX)
Eye positions  : dghs-imgutils detect_eyes on full image, matched to face by containment
Fallback       : insightface 5-pt landmarks on full image, matched by IoU
Last resort    : horizontal bar

Both detectors run on the full image - no crops, no coordinate translation.

Usage:
    uv run main.py input.jpg output.jpg
    uv run main.py input.jpg output.jpg --color 30,0,0
    uv run main.py input.jpg output.jpg --eye-top 0.28 --eye-bot 0.60
"""

import sys
import warnings
import argparse
import functools
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

_verbose = False


def _vprint(*args, **kwargs):
    if _verbose:
        print(*args, **kwargs)

import cv2
import numpy as np
from PIL import Image
from imgutils.detect.face import detect_faces

try:
    from imgutils.detect.eye import detect_eyes as _imgutils_eyes
    _HAS_EYE = True
except Exception:
    _HAS_EYE = False


# ---------------------------------------------------------------------------
# insightface - lazy singleton
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _if_app():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        return app
    except Exception as e:
        _vprint(f"insightface unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Eye matching - run once on the full image, match to each face by containment
# ---------------------------------------------------------------------------

def _eye_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _imgutils_eye_map(pil_img):
    """Return all detected eyes as list of (cx, cy) from imgutils, or []."""
    if not _HAS_EYE:
        return []
    try:
        return [_eye_center(e[0]) for e in _imgutils_eyes(pil_img)]
    except Exception:
        return []


def _insightface_faces(img_bgr):
    app = _if_app()
    return app.get(img_bgr) if app else []


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _eyes_for_face(fx0, fy0, fx1, fy1, all_eye_centers, if_faces):
    """
    Return (left_eye, right_eye) as (x,y) ndarrays, or None.

    Strategy 1: imgutils eye detector - keep any eyes whose center lies inside
                the face bbox, sort by x, take leftmost and rightmost.
    Strategy 2: insightface - find the detection with best IoU overlap and use
                its kps[0]/kps[1] landmarks directly.
    """
    # --- strategy 1 ---
    inside = [(cx, cy) for cx, cy in all_eye_centers
              if fx0 <= cx <= fx1 and fy0 <= cy <= fy1]

    if len(inside) >= 2:
        inside.sort(key=lambda p: p[0])
        return np.array(inside[0], dtype=float), np.array(inside[-1], dtype=float)

    if len(inside) == 1:
        # One eye hidden (hair/occlusion) - mirror across face centre so the bar
        # uses the real eye's Y while still spanning the full face width.
        ex, ey = inside[0]
        mirrored = (fx0 + fx1 - ex, ey)
        pair = sorted([inside[0], mirrored], key=lambda p: p[0])
        return np.array(pair[0], dtype=float), np.array(pair[1], dtype=float)

    # --- strategy 2 ---
    best_iou, best_face = 0.25, None
    for f in if_faces:
        bx0, by0, bx1, by1 = (int(v) for v in f.bbox)
        score = _iou((fx0, fy0, fx1, fy1), (bx0, by0, bx1, by1))
        if score > best_iou:
            best_iou, best_face = score, f

    if best_face is not None and best_face.kps is not None:
        le = np.array(best_face.kps[0], dtype=float)
        re = np.array(best_face.kps[1], dtype=float)
        if le[0] > re[0]:          # insightface uses subject-perspective
            le, re = re, le
        return le, re

    return None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _solid_bar(img, cx, cy, w, h, angle_deg, color):
    box = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
    cv2.fillPoly(img, [np.intp(box)], color)


def _marker_bar(img, cx, cy, w, h, angle_deg, color, n_strokes=4):
    """Fill the bar as n_strokes overlapping ellipses (circular marker motion)."""
    rng = np.random.default_rng()
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    for _ in range(n_strokes):
        lx = rng.uniform(-w * 0.10, w * 0.10)
        ly = rng.uniform(-h * 0.10, h * 0.10)
        gcx = int(cx + lx * cos_a - ly * sin_a)
        gcy = int(cy + lx * sin_a + ly * cos_a)

        axes = (
            max(1, int(w / 2 * rng.uniform(0.88, 1.12))),
            max(1, int(h / 2 * rng.uniform(0.88, 1.12))),
        )
        ell_angle = angle_deg + rng.uniform(-8, 8)

        cv2.ellipse(img, (gcx, gcy), axes, ell_angle, 0, 360, color, -1)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

_EYE_TOP = 0.34
_EYE_BOT = 0.54


def process(img_bgr, eye_top, eye_bot, pad_x, pad_y, bar_color, marker=False, n_strokes=4):
    h, w = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    faces = detect_faces(pil)
    if_faces = None  # populated below, lazily

    if not faces:
        # Anime detector missed it (heavy hair / occlusion) - try insightface
        if_faces = _insightface_faces(img_bgr)
        if if_faces:
            faces = [((int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3])),
                      float(f.det_score), 'face')
                     for f in if_faces]
            _vprint(f"  anime detector missed; insightface found {len(faces)} face(s)")
        else:
            _vprint("  no faces detected")
            return img_bgr
    else:
        _vprint(f"  {len(faces)} face(s) detected")

    # Run eye detectors once on the full image
    all_eyes = _imgutils_eye_map(pil)
    if if_faces is None:
        if_faces = _insightface_faces(img_bgr)
    _vprint(f"  imgutils eyes: {len(all_eyes)}  insightface faces: {len(if_faces)}")

    out = img_bgr.copy()

    for item in faces:
        fx0, fy0, fx1, fy1 = (int(v) for v in item[0])
        fw, fh = fx1 - fx0, fy1 - fy0

        bar_h = fh * (eye_bot - eye_top) + 2 * pad_y
        bar_w = fw + 2 * pad_x

        eyes = _eyes_for_face(fx0, fy0, fx1, fy1, all_eyes, if_faces)

        if eyes is not None:
            le, re = eyes
            center = (le + re) / 2
            # arctan2 in y-down image space: positive = clockwise tilt
            # cv2.boxPoints positive angle = clockwise tilt -> they match, no negation
            angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
            _vprint(f"  face ({fx0},{fy0})-({fx1},{fy1})  angle={angle:+.1f}°"
                    f"  le=({le[0]:.0f},{le[1]:.0f}) re=({re[0]:.0f},{re[1]:.0f})")
        else:
            _vprint(f"  face ({fx0},{fy0})-({fx1},{fy1})  no eyes found -> horizontal")
            center = np.array([(fx0 + fx1) / 2, fy0 + fh * (eye_top + eye_bot) / 2])
            angle = 0.0

        if marker:
            _marker_bar(out, center[0], center[1], bar_w, bar_h, angle, bar_color, n_strokes)
        else:
            _solid_bar(out, center[0], center[1], bar_w, bar_h, angle, bar_color)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def process_file(src, dst, bar_color, eye_top, eye_bot, pad_x, pad_y, marker=False, n_strokes=4):
    img = cv2.imread(str(src))
    if img is None:
        print(f"[skip] cannot read {src}")
        return False

    result = process(img, eye_top=eye_top, eye_bot=eye_bot,
                     pad_x=pad_x, pad_y=pad_y, bar_color=bar_color,
                     marker=marker, n_strokes=n_strokes)

    ok = cv2.imwrite(str(dst), result)
    if not ok:
        print(f"[fail] could not write {dst}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Draw a rotated censor bar aligned with anime character eyes. "
                    "Pass a single file or a directory for batch mode."
    )
    parser.add_argument("input",  help="Image file or input directory")
    parser.add_argument("output", help="Output file or output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detection details")
    parser.add_argument("--marker", action="store_true", help="Draw as overlapping hand-drawn ellipses instead of a solid rectangle")
    parser.add_argument("--strokes", type=int, default=None, help="Number of marker passes; implies --marker (default: 4)")
    parser.add_argument("--color", default="0,0,0", help="Bar colour R,G,B (default: black)")
    parser.add_argument("--pad-x", type=int, default=6)
    parser.add_argument("--pad-y", type=int, default=2)
    parser.add_argument("--eye-top", type=float, default=_EYE_TOP, help=f"Fallback: eye region top fraction (default {_EYE_TOP})")
    parser.add_argument("--eye-bot", type=float, default=_EYE_BOT, help=f"Fallback: eye region bottom fraction (default {_EYE_BOT})")
    args = parser.parse_args()

    global _verbose
    _verbose = args.verbose

    if not _verbose:
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)  # 3 = ERROR, suppresses "Applied providers:"
        except Exception:
            pass

    try:
        r, g, b = (int(v) for v in args.color.split(","))
    except ValueError:
        sys.exit("--color must be R,G,B integers, e.g. 0,0,0")
    bar_color = (b, g, r)

    kwargs = dict(
        bar_color=bar_color,
        eye_top=args.eye_top,
        eye_bot=args.eye_bot,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        marker=args.marker or (args.strokes is not None),
        n_strokes=args.strokes if args.strokes is not None else 4,
    )

    src = Path(args.input)
    dst = Path(args.output)

    if src.is_dir():
        # Batch mode
        files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        if not files:
            sys.exit(f"No images found in {src}")

        dst.mkdir(parents=True, exist_ok=True)
        ok = total = 0
        for f in files:
            total += 1
            print(f"[{total}/{len(files)}] {f.name}")
            if _verbose:
                print()
            if process_file(f, dst / f.name, **kwargs):
                ok += 1

        print(f"Done: {ok}/{total}")
    else:
        # Single-file mode
        if not src.exists():
            sys.exit(f"Input not found: {src}")

        if dst.suffix.lower() not in IMAGE_EXTS:
            # output looks like a directory
            dst.mkdir(parents=True, exist_ok=True)
            dst = dst / src.name

        print(src.name)
        process_file(src, dst, **kwargs)


if __name__ == "__main__":
    main()
