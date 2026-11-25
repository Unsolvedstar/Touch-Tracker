"""Writes a synthetic 'analysis' (runs/_demo) so the UI can be exercised without real footage."""
import json
from pathlib import Path
import cv2, numpy as np, pandas as pd

rng = np.random.default_rng(1)
out = Path(__file__).parent / "runs" / "_demo"; (out / "crops").mkdir(parents=True, exist_ok=True)
fps, stride, W, H, secs = 25.0, 3, 1280, 720, 40
dt = stride / fps
frames = np.arange(0, int(secs * fps), stride)
t = frames / fps
home_x = np.linspace(250, 1000, 6); home_y = [300, 420, 540, 360, 480, 600]
rows = []
def add(tid, cx, cy, kit, sel=slice(None)):
    h = 90 + (cy - 300) * 0.1
    for f, tt, x, y, hh in zip(frames[sel], t[sel], cx, cy, h):
        rows.append((f, tt, tid, x - 22, y - hh, x + 22, y, 0.9, *kit))
for i in range(6):  # home, red; player 3 loses its ID at t=20s (becomes 102)
    cx = home_x[i] + np.cumsum(rng.normal(0, 1.5, len(t))); cy = home_y[i] + np.cumsum(rng.normal(0, 1.0, len(t)))
    if i == 2:
        k = frames >= 20 * fps
        add(3, cx[~k], cy[~k], (200, 30, 30), ~k); add(102, cx[k], cy[k], (200, 30, 30), k)
    else:
        add(i + 1, cx, cy, (200, 30, 30))
for i in range(6):  # away, blue
    add(50 + i, np.full(len(t), 300 + i * 120.0) + rng.normal(0, 3, len(t)).cumsum() * 0.3,
        np.full(len(t), 350 + (i % 3) * 120.0), (30, 40, 190))
P = pd.DataFrame(rows, columns=["frame", "t", "track_id", "x1", "y1", "x2", "y2", "conf", "r", "g", "b"])
# The camera pans: everything above is in WORLD pixels; what the "phone" records is shifted by the pan offset.
off = lambda tt: 220 * np.sin(2 * np.pi * np.asarray(tt) / 40.0)
cam = pd.DataFrame([[int(f), 1, 0, off(tt), 0, 1, 0, 0, 0, 1, 1] for f, tt in zip(frames, t)],
                   columns=["frame"] + [f"m{k}" for k in range(9)] + ["ok"])
cam.to_csv(out / "camera.csv", index=False)
P_world = P.copy()
for c in ("x1", "x2"):
    P[c] = P[c] - off(P["t"])
P.to_csv(out / "players.csv", index=False)
P = P_world  # ball is generated below in world coordinates
# ball: hold 1.2 s at player k (small dribble), travel 0.8 s to next; sequence 0..5 repeated
seq = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1] * 3
B = []
for fi, (f, tt) in enumerate(zip(frames, t)):
    cyc = tt / 2.0; k = int(cyc) % len(seq); ph = cyc - int(cyc)
    a = P[(P.frame == f)]
    def pos(j):
        tid = [1, 2, 3, 4, 5, 6][j] if not (j == 2 and tt >= 20) else 102
        r = a[a.track_id == tid].iloc[0]; return (r.x1 + r.x2) / 2 + 6, r.y2 - 4
    if ph < 0.6:
        x, y = pos(seq[k]); x += 12 * np.sin(ph * 40)
    else:
        x0, y0 = pos(seq[k]); x1, y1 = pos(seq[(k + 1) % len(seq)]); u = (ph - 0.6) / 0.4
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    B.append((f, tt, x - 5, y - 5, x + 5, y + 5, 0.8))
Bd = pd.DataFrame(B, columns=["frame", "t", "x1", "y1", "x2", "y2", "conf"])
for c in ("x1", "x2"):
    Bd[c] = Bd[c] - off(Bd["t"])
Bd.to_csv(out / "ball.csv", index=False)
vp = out / "demo.mp4"
vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for _ in range(int(secs * fps)):
    vw.write(np.full((H, W, 3), (50, 130, 50), np.uint8))
vw.release()
(out / "meta.json").write_text(json.dumps(dict(fps=fps, frames=int(secs * fps), width=W, height=H, stride=stride, dt=dt,
                                               model="synthetic", video=str(vp))))
print("demo written", len(P), "player rows,", len(B), "ball rows")
