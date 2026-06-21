# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the script

```bash
# Single image
uv run main.py input.png output.png

# Batch (directory → directory)
uv run main.py ./input/ ./output/
```

`uv` manages the virtualenv automatically from the inline PEP 723 metadata at the top of `main.py`. No manual `pip install` is needed. First run downloads ONNX models from HuggingFace and insightface CDN; subsequent runs use the cache.

## Architecture

Everything lives in `main.py` — a single uv inline script.

**Detection pipeline (in order of preference):**

1. **Face detection (primary)** — `dghs-imgutils` `detect_faces()` runs on the full PIL image. Returns `[((x0,y0,x1,y1), score, label), ...]`. Anime-specific; handles stylised art that generic detectors miss.

2. **Face detection (fallback)** — if `detect_faces()` returns nothing (e.g. heavy hair occlusion, tight crop), `insightface` `FaceAnalysis(name="buffalo_sc")` is used instead. Its `.bbox` is converted to the same `((x0,y0,x1,y1), score, 'face')` format. The insightface results are reused for eye landmarks (not called twice).

3. **Eye positions (primary)** — `imgutils.detect.eye.detect_eyes()` runs on the full PIL image, returning all eye bboxes. Each face's eyes are matched by checking whether the eye center falls inside the face bbox (`fx0 <= cx <= fx1`). Leftmost and rightmost matched eyes become `le` / `re`. If exactly one eye is found, the missing eye is synthesised by mirroring across the face horizontal centre so the bar still spans the full width at the correct height.

4. **Eye positions (fallback)** — `insightface` `FaceAnalysis(name="buffalo_sc")` runs on the full BGR image. Its detections are matched to each face by IoU (threshold 0.25). `kps[0]`/`kps[1]` give left/right eye centers; x-coordinates are swapped if needed because insightface uses subject-perspective ordering.

5. **Last resort** — horizontal bar centred at `eye_top`/`eye_bot` fractions of face height.

**Both eye detectors run once per image** (not per face), then results are matched. This avoids coordinate-translation bugs and wrong-face matches in group shots.

**Rotation:** `angle = arctan2(re.y - le.y, re.x - le.x)` in y-down image space. Passed directly to `cv2.boxPoints` — no negation. Both conventions agree: positive = clockwise tilt.

**Drawing:** Two modes controlled by `--marker`:

- **Solid (default)** — `_solid_bar`: `cv2.boxPoints` + `cv2.fillPoly`, clean filled rectangle.
- **Marker (`--marker`)** — `_marker_bar`: `n_strokes` overlapping filled ellipses, each fitted to the bar's dimensions with a slightly jittered centre (±10 % of bar size in local coordinates), randomised axes (±12 %), and a small random tilt on top of the bar's rotation angle. Produces curved, non-rectangular edges. Passing `--strokes` implies `--marker`.

## Key parameters

| Flag | Default | Effect |
|---|---|---|
| `--eye-top` | 0.34 | Top of bar as fraction of face height (fallback only) |
| `--eye-bot` | 0.54 | Bottom of bar as fraction of face height (fallback only) |
| `--pad-x` | 6 | Extra pixels left/right beyond face bbox |
| `--pad-y` | 2 | Extra pixels above/below eye region |
| `--color` | `0,0,0` | Bar colour as `R,G,B` |
| `--marker` | off | Draw as overlapping hand-drawn ellipses instead of a solid rectangle |
| `--strokes` | 4 | Number of marker passes; requires `--marker` |
| `--verbose` / `-v` | off | Print face/eye detection details per image |

## Noise suppression

Several third-party warnings are suppressed unconditionally or in non-verbose mode:

- `RuntimeWarning` — blanket filter via `warnings.filterwarnings`
- `FutureWarning` from `insightface` — `face_align.py` uses deprecated `tform.estimate()`; fix lives in the library, not here
- HF Hub unauthenticated request warning — suppressed via `logging.getLogger("huggingface_hub").setLevel(ERROR)` when not verbose
- ONNX Runtime "Applied providers:" — suppressed via `ort.set_default_logger_severity(3)` when not verbose

## Dependencies

Declared inline in the script header (PEP 723). Changing them triggers uv to rebuild the environment:

- `dghs-imgutils` — anime-specific detection (ONNX via `onnxruntime`)
- `onnxruntime` — must be explicit; imgutils tries to `pip install` it at runtime otherwise, which fails in uv environments (no pip)
- `insightface` — face detection fallback + 5-point landmark fallback
- `opencv-python>=4.9`, `numpy>=2.0` — `numpy>=2.0` avoids MINGW-W64 float128 warnings from uv's bundled Python on Windows
