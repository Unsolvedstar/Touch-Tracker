"""Builds a labelled small-ball training set from YOUR OWN footage, with no outside data.

Real footballs filmed by this camera (sprites cut from the video) are shrunk to match-ball size and pasted,
with blur / motion blur / shadow / noise, onto real frames of the same video, next to real players.
Backgrounds come from times OUTSIDE the evaluation window, so recall measured there is on unseen frames.

    python make_ball_dataset.py VIDEO OUT_DIR --eval-start 120 --eval-end 240 --sprites ballkit/sprites

Output: YOLO dataset (images/ labels/ train+val, dataset.yaml), one class "ball", 640x640 tiles at native scale.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

TILE = 640


def load_sprites(folder: Path) -> list[np.ndarray]:
    """BGRA circular cut-outs from the raw square crops (ball fills the central ~77%)."""
    out = []
    for f in sorted(Path(folder).glob("raw_*.png")):
        im = cv2.imread(str(f))
        s = im.shape[0]
        d = int(round(s / 1.3))
        o = (s - d) // 2
        ball = im[o:o + d, o:o + d]
        yy, xx = np.mgrid[:d, :d]
        r = np.hypot(xx - d / 2 + 0.5, yy - d / 2 + 0.5)
        alpha = np.clip((d / 2 * 0.96 - r) / 2.0, 0, 1)  # soft 2px edge
        out.append(np.dstack([ball, (alpha * 255).astype(np.uint8)]))
    return out


def motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, M, (length, length))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def procedural_ball(rng: random.Random, size: int = 64) -> np.ndarray:
    """A generic sunlit football: bright disk, soft grey underside, a few dark panel patches."""
    base = rng.uniform(205, 250)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = size / 2
    r = np.hypot(xx - c, yy - c) / c
    shade = 1 - 0.25 * np.clip((yy - c) / c, 0, 1) - 0.1 * (xx > c)          # darker underside / right
    img = np.clip(base * shade, 0, 255)[:, :, None].repeat(3, axis=2)
    for _ in range(rng.randint(1, 3)):                                          # dark panel patches
        cv2.circle(img, (int(c + rng.uniform(-0.45, 0.45) * c), int(c + rng.uniform(-0.45, 0.45) * c)),
                   int(rng.uniform(0.15, 0.3) * c), (rng.uniform(40, 110),) * 3, -1)
    img[:, :, 0] *= rng.uniform(0.95, 1.05); img[:, :, 2] *= rng.uniform(0.95, 1.05)
    alpha = np.clip((0.97 - r) * c / 2.0, 0, 1)
    return np.dstack([np.clip(img, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])


def make_sprite(base: np.ndarray, d: int, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.35:
        base = procedural_ball(rng)
    s = base.copy()
    s = cv2.warpAffine(s, cv2.getRotationMatrix2D((s.shape[1] / 2, s.shape[0] / 2), rng.uniform(0, 360), 1.0),
                       (s.shape[1], s.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        s = s[:, ::-1]
    bgr = s[:, :, :3].astype(np.float32)
    bgr = bgr * rng.uniform(0.95, 1.4) + rng.uniform(-5, 25)                      # exposure (sunlit, bright)
    gray = bgr.mean(axis=2, keepdims=True)
    bgr = gray + (bgr - gray) * rng.uniform(0.6, 1.4)                             # saturation
    if rng.random() < 0.3:                                                         # a tinted ball (yellow / orange / pink)
        tint = np.array([rng.uniform(0.7, 1.0), rng.uniform(0.8, 1.05), rng.uniform(0.9, 1.2)])
        bgr = bgr * tint
    s = np.dstack([np.clip(bgr, 0, 255).astype(np.uint8), s[:, :, 3]])
    s = cv2.resize(s, (d, d), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.4 and d >= 6:                                              # moving ball
        s = motion_blur(s, rng.randint(2, max(3, d // 2 + 2)), rng.uniform(0, 180))
    sigma = rng.uniform(0.3, 0.9)
    return cv2.GaussianBlur(s, (0, 0), sigma)


def paste(img: np.ndarray, sprite: np.ndarray, cx: float, cy: float, rng: random.Random) -> None:
    d = sprite.shape[0]
    H, W = img.shape[:2]
    x0, y0 = int(round(cx - d / 2)), int(round(cy - d / 2))
    if x0 < 2 or y0 < 2 or x0 + d > W - 2 or y0 + d > H - 2:
        return
    # soft shadow on the ground, slightly down-right of the ball
    sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (int(cx + 0.1 * d), int(cy + 0.42 * d)), (max(int(0.5 * d), 1), max(int(0.17 * d), 1)), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(d / 8, 0.6)) * rng.uniform(0.25, 0.6)
    img[:] = (img.astype(np.float32) * (1 - sh[:, :, None])).astype(np.uint8)
    a = sprite[:, :, 3:4].astype(np.float32) / 255
    roi = img[y0:y0 + d, x0:x0 + d].astype(np.float32)
    img[y0:y0 + d, x0:x0 + d] = (roi * (1 - a) + sprite[:, :, :3] * a).astype(np.uint8)


def read_frame(cap, t: float, fps: float):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    return f if ok else None


def person_boxes(model, frame) -> np.ndarray:
    r = model.predict(frame, imgsz=960, classes=[0], conf=0.3, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--sprites", default="ballkit/sprites")
    ap.add_argument("--eval-start", type=float, default=120); ap.add_argument("--eval-end", type=float, default=240)
    ap.add_argument("--frames-train", type=int, default=450); ap.add_argument("--frames-val", type=int, default=60)
    ap.add_argument("--tiles-per-frame", type=int, default=3)
    ap.add_argument("--height-fit", default=None, help="CSV of players (y1,y2) to fit player height vs image row")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from ultralytics import YOLO

    rng = random.Random(a.seed)
    out = Path(a.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True); (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    sprites = load_sprites(Path(a.sprites))
    # sprites 0-6 are clean single balls; later ones can include a neighbouring ball at the edge
    sprites = sprites[:7] if len(sprites) >= 7 else sprites
    assert sprites, "no sprites found"

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    times = [t for t in np.arange(3, dur - 3, 0.5) if not (a.eval_start - 3 <= t <= a.eval_end + 3)]
    rng.shuffle(times)
    train_t, val_t = times[:a.frames_train], times[a.frames_train:a.frames_train + a.frames_val]
    print(f"{len(train_t)} train frames, {len(val_t)} val frames (eval window {a.eval_start}-{a.eval_end}s excluded)")

    # player height as a function of image row, to size the ball realistically (ball ~ 1/8 of a player's height)
    import pandas as pd
    P = pd.read_csv(a.height_fit) if a.height_fit else None
    coef = np.polyfit(P["y2"], P["y2"] - P["y1"], 1) if P is not None else np.array([0.2, -30.0])
    det = YOLO("yolov8n.pt")

    n_img = 0
    for split, tl in (("train", train_t), ("val", val_t)):
        for fi, t in enumerate(tl):
            frame = read_frame(cap, float(t), fps)
            if frame is None:
                continue
            boxes = person_boxes(det, frame)
            H, W = frame.shape[:2]
            for k in range(a.tiles_per_frame):
                img = frame.copy()
                labels = []
                if rng.random() > 0.10:                                           # ~10% pure negatives
                    for _ in range(1 if rng.random() < 0.85 else 2):
                        if len(boxes) and rng.random() < 0.8:                     # next to a real player's feet
                            b = boxes[rng.randrange(len(boxes))]
                            h = b[3] - b[1]
                            cx = (b[0] + b[2]) / 2 + rng.uniform(-0.45, 0.45) * h
                            cy = b[3] + rng.uniform(-0.25, 0.08) * h
                            if rng.random() < 0.15:                                # in the air
                                cy -= rng.uniform(0.3, 1.0) * h
                        else:
                            cy = rng.uniform(0.42 * H, 0.9 * H); cx = rng.uniform(0.05 * W, 0.95 * W)
                            h = max(np.polyval(coef, cy), 30)
                        d = int(round(np.clip(h * rng.uniform(0.10, 0.15), 5, 28)))
                        sp = make_sprite(rng.choice(sprites), d, rng)
                        paste(img, sp, cx, cy, rng)
                        labels.append((cx, cy, d))
                # tile that contains the (first) ball, or a random player-band tile for negatives
                if labels:
                    bx, by, _ = labels[0]
                else:
                    bx, by = rng.uniform(0.2 * W, 0.8 * W), rng.uniform(0.5 * H, 0.85 * H)
                x0 = int(np.clip(bx - rng.uniform(0.15, 0.85) * TILE, 0, W - TILE))
                y0 = int(np.clip(by - rng.uniform(0.15, 0.85) * TILE, 0, H - TILE)) if H >= TILE else 0
                tile = img[y0:y0 + TILE, x0:x0 + TILE]
                if tile.shape[0] != TILE or tile.shape[1] != TILE:
                    continue
                q = rng.randint(60, 95)                                            # compression + sensor noise
                tile = cv2.imdecode(cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
                name = f"{split}_{n_img:06d}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), tile)
                lines = []
                for cx, cy, d in labels:
                    lx, ly = cx - x0, cy - y0
                    if 0 < lx < TILE and 0 < ly < TILE:
                        s = d + 2
                        lines.append(f"0 {lx / TILE:.6f} {ly / TILE:.6f} {s / TILE:.6f} {s / TILE:.6f}")
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                n_img += 1
            if fi % 50 == 0:
                print(f"  {split}: {fi}/{len(tl)} frames, {n_img} tiles", flush=True)
    (out / "dataset.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    print("done:", n_img, "tiles ->", out)


if __name__ == "__main__":
    main()
