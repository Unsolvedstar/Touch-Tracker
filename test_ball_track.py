"""Trajectory linking: recover a ball seen in 35% of frames among heavy clutter, vs taking the best-confidence box."""
import numpy as np, pandas as pd
from ball_track import link_ball

rng = np.random.default_rng(3)
N, DT, VH = 400, 0.133, 80.0
frames = np.arange(N) * 4
t = np.arange(N) * DT
# true ball: a series of passes between waypoints (speed ~ 5 player-heights/s)
way = rng.uniform([200, 300], [1800, 700], size=(40, 2))
pos = np.zeros((N, 2)); k = 0; cur = way[0].copy(); spd = 5.0
for i in range(N):
    tgt = way[(k + 1) % len(way)]; v = tgt - cur; dist = np.linalg.norm(v)
    if i % 40 == 0: spd = rng.uniform(1.5, 12.0)          # each "pass" has its own speed (player-heights/s)
    stepd = spd * VH * DT
    if dist < stepd: k += 1; cur = tgt.copy()
    else: cur = cur + v / dist * stepd
    pos[i] = cur
# players: 10 jogging around the ball's neighbourhood
P = []
for f, tt, b in zip(frames, t, pos):
    for pid in range(10):
        off = np.array([np.cos(pid + tt * 0.3), np.sin(pid * 1.7 + tt * 0.2)]) * 260
        P.append((f, tt, *(b + off), VH))
players = pd.DataFrame(P, columns=["frame", "t", "sfx", "sfy", "sh"])
# candidates: ball detected 35% of frames (conf 0.15-0.6) + 3 clutter boxes per frame (conf 0.05-0.4), some near players
rows, truth = [], set()
for i, (f, tt, b) in enumerate(zip(frames, t, pos)):
    if rng.random() < 0.35:
        rows.append((f, tt, *(b + rng.normal(0, 2, 2)), rng.uniform(0.15, 0.6))); truth.add((f, len(rows) - 1))
    for _ in range(3):
        c = players[players.frame == f].iloc[rng.integers(10)]
        near = rng.random() < 0.5
        xy = np.array([c.sfx, c.sfy]) + rng.normal(0, 60, 2) if near else rng.uniform([0, 0], [2000, 1000])
        rows.append((f, tt, *xy, rng.uniform(0.05, 0.4)))
C = pd.DataFrame(rows, columns=["frame", "t", "sx", "sy", "conf"])
for a, b in (("x1", "sx"), ("x2", "sx")): C[a] = C[b] - 5
for a, b in (("y1", "sy"), ("y2", "sy")): C[a] = C[b] - 5
is_true = np.zeros(len(C), bool); is_true[[i for _, i in truth]] = True
C["true"] = is_true

def score(sel):
    tp = int(sel["true"].sum()); return tp, len(sel) - tp
base = C.sort_values("conf", ascending=False).drop_duplicates("frame")
chosen = link_ball(C, players)
n_true = int(C["true"].sum())
for name, sel in (("best-confidence per frame", base), ("trajectory linking", chosen)):
    tp, fp = score(sel); print(f"{name:26s}: found {tp}/{n_true} true ({tp/n_true:.0%}) | false {fp} | precision {tp/max(tp+fp,1):.0%}")
tp, fp = score(chosen); btp, bfp = score(base)
f1 = lambda tp, fp: 2 * (tp / n_true) * (tp / max(tp + fp, 1)) / max(tp / n_true + tp / max(tp + fp, 1), 1e-9)
assert f1(tp, fp) > 1.6 * f1(btp, bfp) and tp / n_true > 0.6 and tp / max(tp + fp, 1) > 0.6, (tp, fp, n_true)
assert chosen.frame.is_unique
print("OK")
