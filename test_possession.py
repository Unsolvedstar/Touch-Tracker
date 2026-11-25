"""Synthetic check: A dribbles with 3 contacts, passes to B, B holds then receives nothing else."""
import numpy as np
import pandas as pd

from possession import PossessionParams, compute_possession

DT = 0.1
N = 100  # 10 s
H = 100  # player height px
frames = np.arange(N)


def player(tid, xs, y=500):
    return pd.DataFrame({"frame": frames, "t": frames * DT, "track_id": tid,
                         "x1": xs - 20, "x2": xs + 20, "y1": y - H, "y2": y})


# Player 1 stationary at x=200; player 2 stationary at x=800; player 3 (opposition) far away.
p = pd.concat([player(1, np.full(N, 200.0)), player(2, np.full(N, 800.0)), player(3, np.full(N, 1500.0))])

# Ball: 0-3s near P1 with 3 kicks (direction flips each second), 3-5s rolls to P2, 5-8s held by P2, then lost
bx = np.zeros(N); by = np.full(N, 495.0)
# three sharp kicks (20 px/frame for 2 frames, alternating direction) while staying within the control radius
kick = np.array([0,20,40,40,40,40,40,40,40,40])
seg = np.concatenate([kick, 40 - kick, kick])
bx[:30] = 200 + seg
bx[30:50] = np.linspace(bx[29], 790, 20)
bx[50:80] = 800 + np.sin(np.linspace(0, 6, 30)) * 5
bx[80:] = np.nan
ball = pd.DataFrame({"frame": frames, "t": frames * DT, "x1": bx - 5, "x2": bx + 5, "y1": by - 5, "y2": by + 5, "conf": 0.9}).dropna()

summary, events, touches = compute_possession(p, ball, {1: "#1 A", 2: "#2 B"}, DT, PossessionParams(touch_dv_h_s=1.0))
print(summary, "\n", events, sep="\n")

s = summary.set_index("player")
assert set(s.index) == {"#1 A", "#2 B"}, s
assert s.loc["#1 A", "touches"] >= 2, "dribble contacts should add touches"
assert s.loc["#2 B", "touches"] >= 1
assert 2.0 <= s.loc["#1 A", "time_on_ball_s"] <= 4.0, s
assert 2.5 <= s.loc["#2 B", "time_on_ball_s"] <= 4.0, s
assert len(touches) == int(summary["touches"].sum()) and touches[["x", "y"]].notna().all().all()
print("OK")
