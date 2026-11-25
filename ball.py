"""Small-ball search. A ball in wide phone footage is ~10 px across, which a detector run on the whole
downscaled frame misses. So:
  * only search where the ball can be: the area around the players (it is always near play);
  * run the detector on overlapping tiles at NATIVE resolution (no downscaling);
  * keep several candidates per frame (low confidence threshold); ball_track.py decides which are real
    by following them through time.

Uses custom weights (models/ball.pt, single class) if present, otherwise the COCO 'sports ball' class.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from library import find_video

TILE_W, TILE_H = 800, 600
COCO_BALL = 32
MAX_CANDIDATES = 6
CUSTOM_WEIGHTS = Path(__file__).parent / "models" / "ball.pt"
COLS = ["frame", "t", "x1", "y1", "x2", "y2", "conf"]


def search_roi(boxes: np.ndarray, W: int, H: int, margin: float = 0.7):
    """Rectangle around all players, grown by `margin` x the median player height (ball can be ahead of everyone)."""
    if len(boxes) == 0:
        return None
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    m = max(margin * h, 40)
    return (max(boxes[:, 0].min() - m, 0), max(boxes[:, 1].min() - m, 0),
            min(boxes[:, 2].max() + m, W), min(boxes[:, 3].max() + m, H))


def tile_origins(roi, W: int, H: int, tw: int = TILE_W, th: int = TILE_H):
    """Top-left corners of overlapping tiles (each fully inside the frame) that cover the ROI."""
    x1, y1, x2, y2 = roi

    def axis(lo, hi, size, limit):
        if limit <= size:
            return [0]
        n = max(int(np.ceil((hi - lo - size * 0.25) / (size * 0.75))), 1)
        starts = np.linspace(lo, max(hi - size, lo), n) if hi - lo > size else [(lo + hi) / 2 - size / 2]
        return sorted({int(np.clip(s, 0, limit - size)) for s in starts})

    return [(x, y) for y in axis(y1, y2, th, H) for x in axis(x1, x2, tw, W)]


def merge_duplicates(cands: list[tuple]) -> list[tuple]:
    """Tiles overlap, so one ball can be found twice. Keep the most confident of any pair that nearly coincide."""
    out: list[tuple] = []
    for c in sorted(cands, key=lambda c: -c[4]):
        cx, cy, size = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2, max(c[2] - c[0], c[3] - c[1])
        if all(np.hypot(cx - (o[0] + o[2]) / 2, cy - (o[1] + o[3]) / 2) > 0.6 * max(size, 8) for o in out):
            out.append(c)
    return out


def detect_in_frame(model, frame: np.ndarray, player_boxes: np.ndarray, cls: list[int], conf: float = 0.05,
                    imgsz: int = TILE_W) -> list[tuple]:
    H, W = frame.shape[:2]
    roi = search_roi(player_boxes, W, H)
    if roi is None:
        return []
    ph = player_boxes[:, 3] - player_boxes[:, 1]
    max_px = max(0.22 * float(np.quantile(ph, 0.9)), 12)  # ball ~1/8 of a player's height; allow slack
    found: list[tuple] = []
    for x0, y0 in tile_origins(roi, W, H):
        r = model.predict(frame[y0:y0 + TILE_H, x0:x0 + TILE_W], imgsz=imgsz, classes=cls, conf=conf, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            w, h = b[2] - b[0], b[3] - b[1]
            if 3 <= max(w, h) <= max_px:
                found.append((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, float(c)))
    return merge_duplicates(found)[:MAX_CANDIDATES]


def run_ball_pass(run_dir: Path, model_name: str = "yolov8s.pt", progress=None, max_frames: int | None = None) -> Path:
    """Search every analysed frame of an existing run for ball candidates -> ball_candidates.csv."""
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    players = pd.read_csv(run_dir / "players.csv")
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in players.groupby("frame")}
    frames = sorted(by_frame)
    if max_frames:
        frames = frames[:max_frames]
    wanted = set(frames)
    if CUSTOM_WEIGHTS.exists():
        model, cls = YOLO(str(CUSTOM_WEIGHTS)), [0]
    else:
        model, cls = YOLO(model_name), [COCO_BALL]
    t_of = players.drop_duplicates("frame").set_index("frame")["t"]

    cap = cv2.VideoCapture(find_video(meta["video"], run_dir))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {meta['video']}")
    first = frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows, i = [], first - 1
    done = 0
    while done < len(frames) and cap.grab():
        i += 1
        if i not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        for (x1, y1, x2, y2, c) in detect_in_frame(model, frame, by_frame[i], cls):
            rows.append((i, float(t_of[i]), x1, y1, x2, y2, c))
        done += 1
        if progress:
            progress(done / len(frames))
    cap.release()
    out = run_dir / "ball_candidates.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)
    (run_dir / "ball_search.json").write_text(json.dumps({"weights": "custom" if CUSTOM_WEIGHTS.exists() else model_name}))
    return out
