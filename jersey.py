"""Read shirt numbers from saved player crops (EasyOCR) and vote per track.

Phone footage gives mostly tiny, blurred crops, so reading every crop produces confident nonsense. Instead:
  * the saved crops are already torso-only (detect.py keeps each track's 6 largest), so they're read whole,
  * each crop casts ONE vote (its best reading), and
  * a track only gets a number when at least `min_votes` crops agree on it.
`votes` / `crops_read` are reported alongside the confidence so the app can show how much evidence there is.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_reader = None
MIN_SIDE = 28  # px; crops smaller than this (height) can't hold a readable digit


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # heavy import, load lazily

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _variants(img: np.ndarray):
    scale = max(3, int(round(120 / max(img.shape[0], 1))))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    yield img
    yield clahe
    yield 255 - clahe  # dark digits on light shirts


def _read_crop(reader, img: np.ndarray) -> tuple[int, float] | None:
    """Best (number, confidence) in one crop, or None. Variants must not be allowed to vote several times."""
    best: dict[int, float] = defaultdict(float)
    for v in _variants(img):
        for _, text, conf in reader.readtext(v, allowlist="0123456789", detail=1):
            text = text.strip()
            if 1 <= len(text) <= 2 and conf >= 0.3:
                best[int(text)] = max(best[int(text)], conf)  # same number in several variants = still one vote
    if not best:
        return None
    n = max(best, key=best.get)
    return n, best[n]


def read_numbers(crop_dir: Path, track_ids, progress=None, min_votes: int = 2) -> dict:
    """-> {track_id: (number or None, confidence 0-1, votes, crops_read)}.
    confidence = (votes for the winner / crops that produced any reading), shrunk when evidence is thin."""
    reader = _get_reader()
    out: dict = {}
    track_ids = list(track_ids)
    for n, tid in enumerate(track_ids):
        votes: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        read = 0
        for path in sorted(Path(crop_dir).glob(f"{tid}_*.jpg")):
            img = cv2.imread(str(path))
            if img is None or img.shape[0] < MIN_SIDE:
                continue
            r = _read_crop(reader, img)
            if r:
                read += 1
                counts[r[0]] += 1
                votes[r[0]] += r[1]
        if counts:
            num = max(counts, key=lambda k: (counts[k], votes[k]))
            k = counts[num]
            if k >= min_votes:
                out[tid] = (num, round(k / (read + 1), 2), k, read)  # +1: one unexplained crop's worth of doubt
            else:
                out[tid] = (None, 0.0, k, read)
        else:
            out[tid] = (None, 0.0, 0, 0)
        if progress:
            progress((n + 1) / len(track_ids))
    return out
