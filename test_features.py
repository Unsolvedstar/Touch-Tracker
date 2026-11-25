"""Synthetic checks for track stitching, passes, pitch geometry and video rendering."""
import json, tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from passes import detect_passes, pass_matrix, pass_summary
from pitch import homography, movement_stats, player_positions, to_pitch
from possession import OTHER
from render import render_annotated
from tracks import stitch_tracks

DT = 0.1


def box(frames, tid, x, y=500, h=100):
    return pd.DataFrame({"frame": frames, "t": frames * DT, "track_id": tid, "x1": x - 20, "x2": x + 20, "y1": y - h, "y2": y})


# ---- stitching: track 1 ends at x=300 (t=5), track 2 starts 0.6s later at x=330, track 3 starts far away
f1, f2, f3 = np.arange(0, 50), np.arange(56, 100), np.arange(56, 100)
P = pd.concat([box(f1, 1, np.full(50, 300.0)), box(f2, 2, np.full(44, 330.0)), box(f3, 3, np.full(44, 1500.0))])
roots = stitch_tracks(P, [1, 2, 3], DT)
assert roots[2] == 1 and roots[1] == 1 and roots[3] == 3, roots
# overlapping tracks must never merge
Q = pd.concat([box(np.arange(0, 50), 1, np.full(50, 300.0)), box(np.arange(10, 60), 2, np.full(50, 320.0))])
assert stitch_tracks(Q, [1, 2], DT) == {1: 1, 2: 2}

# ---- passes
ev = pd.DataFrame([
    dict(player="A", start_t=0, end_t=2), dict(player="B", start_t=2.3, end_t=4),   # A->B completed
    dict(player="B", start_t=4.1, end_t=5), dict(player=OTHER, start_t=5.4, end_t=7),  # B loses ball
    dict(player=OTHER, start_t=7.1, end_t=8), dict(player="A", start_t=8.4, end_t=9),  # A wins ball
    dict(player="A", start_t=20, end_t=21), dict(player="B", start_t=40, end_t=41),    # 19 s gap: no pass
])
ps = detect_passes(ev)
assert list(ps["outcome"]) == ["completed", "lost", "won"], ps
s = pass_summary(ps, ["A", "B"]).set_index("player")
assert s.loc["A", "passes_completed"] == 1 and s.loc["B", "ball_lost"] == 1 and s.loc["A", "ball_won"] == 1
m = pass_matrix(ps, ["A", "B"])
assert m.loc["A", "B"] == 1 and m.values.sum() == 1

# ---- pitch: image rectangle (200,100)-(1000,700) -> 105 x 68 m; centre of image rect -> centre of pitch
H = homography([(200, 100), (1000, 100), (1000, 700), (200, 700)], 105, 68)
X, Y = to_pitch(H, [600, 200, 1000], [400, 100, 700])
assert np.allclose(X, [52.5, 0, 105], atol=1e-3) and np.allclose(Y, [34, 0, 68], atol=1e-3)
# runner covering 4 px/step along x -> metres; a 5 m teleport (ID swap) must be dropped
xs = 200 + np.arange(50) * 4.0
xs[25:] += 400  # glitch
pp = player_positions(box(np.arange(50), 1, xs), {1: "A"}, H)
ms = movement_stats(pp)
assert 22 <= ms.loc[0, "distance_m"] <= 27, ms  # 48 valid steps x 0.52 m, glitch step dropped

# ---- render: 2 s synthetic video, one home player holding the ball
d = Path(tempfile.mkdtemp())
vp = d / "v.mp4"
vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
for _ in range(50):
    vw.write(np.full((360, 640, 3), (40, 120, 40), np.uint8))
vw.release()
meta = dict(fps=25.0, stride=2, dt=0.08, video=str(vp), frames=50)
fr = np.arange(0, 50, 2)
pl = pd.concat([box(fr, 1, np.full(len(fr), 200.0), 250, 80), box(fr, 9, np.full(len(fr), 400.0), 250, 80)])
bl = pd.DataFrame({"frame": fr, "t": fr / 25, "x1": 215, "x2": 225, "y1": 240, "y2": 250, "conf": 0.9})
evs = pd.DataFrame([dict(player="#7 Test", start_t=0.0, end_t=1.5, duration_s=1.5, touches=2, start_frame=0, end_frame=36)])
tc = pd.DataFrame([dict(player="#7 Test", frame=0, t=0.0, x=220, y=245), dict(player="#7 Test", frame=20, t=0.8, x=220, y=245)])
out, playable = render_annotated(meta, pl, bl, {1: "#7 Test"}, evs, tc, d / "out.mp4", 0, 2)
cap = cv2.VideoCapture(str(out))
assert out.stat().st_size > 1000 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 45
print("render ok, browser-playable h264:", playable)
print("OK")
