"""Fine-tuning set from hand-checked REAL detections: real ball tiles (positives) and real lookalikes (boots, the
referee's shirt, dry grass) as hard negatives, mixed with a slice of the synthetic set so the model keeps generalising.

    python make_real_ball_set.py VIDEO REVIEW_DIR OUT_DIR SYNTH_DIR
Labels live in REVIEW_DIR/labels.json: {"sources": {"linked": {"csv": ..., "pos": [...], "neg": [...]}, ...}}
(indices refer to rows of that CSV; each row has frame, x1, y1, x2, y2)."""
import json, random, shutil, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

video, review, out, synth = map(Path, sys.argv[1:5])
rng = random.Random(0)
lab = json.loads((review / "labels.json").read_text())
pos, neg = [], []
for name, src in lab["sources"].items():
    df = pd.read_csv(review / src["csv"])
    for i in src["pos"]: r = df.iloc[i]; pos.append((int(r.frame), (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, max(r.x2 - r.x1, r.y2 - r.y1)))
    for i in src["neg"]: r = df.iloc[i]; neg.append((int(r.frame), (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, max(r.x2 - r.x1, r.y2 - r.y1)))
def dedupe(items):
    keep = []
    for f, x, y, s in items:
        if not any(f == g and np.hypot(x - a, y - b) < 8 for g, a, b, _ in keep): keep.append((f, x, y, s))
    return keep
pos, neg = dedupe(pos), dedupe(neg)
pos_frames = {f for f, *_ in pos}
neg = [n for n in neg if n[0] not in pos_frames]      # a negative tile must not hide an unlabelled real ball
print(len(pos), "positives,", len(neg), "negatives")
for s in ("train", "val"):
    (out / "images" / s).mkdir(parents=True, exist_ok=True); (out / "labels" / s).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(video)); TILE = 640
frames = {}
def frame(f):
    if f not in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); frames[f] = im if ok else None
    return frames[f]
n = 0
def emit(f, cx, cy, size, is_pos, copies):
    global n
    im = frame(f)
    if im is None: return
    H, W = im.shape[:2]
    for k in range(copies):
        x0 = int(np.clip(cx - rng.uniform(0.12, 0.88) * TILE, 0, W - TILE)); y0 = int(np.clip(cy - rng.uniform(0.12, 0.88) * TILE, 0, H - TILE))
        tile = im[y0:y0 + TILE, x0:x0 + TILE]
        if tile.shape[:2] != (TILE, TILE): continue
        name = f"real_{n:05d}"; n += 1
        cv2.imwrite(str(out / "images/train" / f"{name}.jpg"), tile)
        s = max(size, 8) + 2
        (out / "labels/train" / f"{name}.txt").write_text(f"0 {(cx-x0)/TILE:.6f} {(cy-y0)/TILE:.6f} {s/TILE:.6f} {s/TILE:.6f}" if is_pos else "")
for f, x, y, s in pos: emit(f, x, y, s, True, 12)
for f, x, y, s in neg: emit(f, x, y, s, False, 6)
# slice of synthetic data (train) + its val for monitoring
syn = sorted((synth / "images/train").glob("*.jpg")); rng.shuffle(syn)
for img in syn[:500]:
    shutil.copy(img, out / "images/train" / img.name); shutil.copy(synth / "labels/train" / (img.stem + ".txt"), out / "labels/train" / (img.stem + ".txt"))
for img in sorted((synth / "images/val").glob("*.jpg"))[:100]:
    shutil.copy(img, out / "images/val" / img.name); shutil.copy(synth / "labels/val" / (img.stem + ".txt"), out / "labels/val" / (img.stem + ".txt"))
(out / "dataset.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
print("tiles:", len(list((out / 'images/train').glob('*.jpg'))), "train,", len(list((out / 'images/val').glob('*.jpg'))), "val ->", out)
