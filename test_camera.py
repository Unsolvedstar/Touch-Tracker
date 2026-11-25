"""Camera-motion compensation, checked against a simulated pinhole camera with a KNOWN focal length that
swings 70 degrees out and 50 back (far more than a single image plane can represent)."""
import cv2
import numpy as np
import pandas as pd

from camera import CameraModel, HCOLS, stabilise
from detect import estimate_motion
from pitch import homography, player_positions
from possession import PossessionParams, compute_possession

# ---- 1. motion estimator recovers a known pan (camera moves right 30 px => scene shifts left)
rng = np.random.default_rng(0)
world = cv2.GaussianBlur((rng.random((900, 2000)) * 255).astype(np.uint8), (0, 0), 2)
H12, n = estimate_motion(world[100:460, 400:1040], world[100:460, 430:1070], [[300, 150, 340, 260]])
assert H12 is not None and n > 50 and abs(H12[0, 2] - 30) < 1.0 and abs(H12[1, 2]) < 1.0, (H12, n)
print("estimator ok:", round(H12[0, 2], 2))

# ---- simulated pinhole camera ------------------------------------------------------------------
W, H, F_TRUE, CAM_H = 1920, 1080, 1700.0, 4.0
K = np.array([[F_TRUE, 0, W / 2], [0, F_TRUE, H / 2], [0, 0, 1.0]])
Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
Rx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
DT, N = 0.1, 100
frames = np.arange(N)
t = frames * DT
s = np.clip((t - 2) / 3, 0, 1)                     # pan eases in at 2 s, out at 5 s ...
s2 = np.clip((t - 6) / 3, 0, 1)                    # ... and back 50 degrees between 6 s and 9 s
yaw = np.radians(70) * (s * s * (3 - 2 * s)) - np.radians(50) * (s2 * s2 * (3 - 2 * s2))
R_cam = [Ry(a) @ Rx(np.radians(12)) for a in yaw]   # camera -> world; looking slightly down


def project(P, R):
    """World point (x, y-down, z) -> pixel for a camera with orientation R at the origin."""
    v = K @ (R.T @ np.asarray(P, float))
    return v[:2] / v[2]


def step_table(ok=1):
    rows = [[0, *np.eye(3).ravel(), 1]]
    for i in range(1, N):
        Hs = K @ R_cam[i - 1].T @ R_cam[i] @ np.linalg.inv(K)     # current pixels -> previous pixels
        rows.append([i, *(Hs / Hs[2, 2]).ravel(), ok])
    return pd.DataFrame(rows, columns=["frame"] + HCOLS + ["ok"])


table = step_table()

# ---- 2. focal length is recovered from the footage alone
cam = CameraModel(table, W, H)
print(f"focal estimate {cam.f:.0f} (true {F_TRUE:.0f}); swing {cam.pan_degrees():.1f} deg")
assert abs(cam.f - F_TRUE) / F_TRUE < 0.03, cam.f
true_swing = max(np.degrees(np.arccos(np.clip(R[:, 2] @ R_cam[0][:, 2], -1, 1))) for R in R_cam)
assert abs(cam.pan_degrees() - true_swing) < 1.0, (cam.pan_degrees(), true_swing)

# ---- 3. a stationary player has a constant stabilised position while the camera swings 70 degrees and back
def person(ground_xz, tid):
    X, Z = ground_xz
    feet = np.array([project((X, CAM_H, Z), R) for R in R_cam]); head = np.array([project((X, CAM_H - 1.8, Z), R) for R in R_cam])
    return pd.DataFrame({"frame": frames, "t": t, "track_id": tid, "x1": feet[:, 0] - 20, "x2": feet[:, 0] + 20,
                         "y1": head[:, 1], "y2": feet[:, 1]})


p1 = person((8.0, 35.0), 1)
ps, _ = stabilise(p1, pd.DataFrame(columns=["frame", "t", "x1", "y1", "x2", "y2", "conf"]), cam)
raw_range = np.ptp(p1[["x1", "x2"]].mean(axis=1))
print(f"raw foot x moved {raw_range:.0f}px; stabilised wobble {np.ptp(ps['sfx']):.2f}px")
assert raw_range > 1500 and np.ptp(ps["sfx"]) < 15 and np.ptp(ps["sfy"]) < 15, (raw_range, np.ptp(ps["sfx"]))
assert ps["sh"].std() / ps["sh"].mean() < 0.05

# ---- 4. ball resting at a player's feet while the camera pans: no phantom touches once stabilised
foot = np.array([8.0, CAM_H, 35.0]); ball_w = foot + [0.3, -0.1, 0.0]
bpix = np.array([project(ball_w, R) for R in R_cam])
ball = pd.DataFrame({"frame": frames, "t": t, "x1": bpix[:, 0] - 5, "x2": bpix[:, 0] + 5, "y1": bpix[:, 1] - 5,
                     "y2": bpix[:, 1] + 5, "conf": 0.9})
params = PossessionParams()
_, _, raw_t = compute_possession(p1, ball, {1: "#1"}, DT, params)             # no camera information
ps, bs = stabilise(p1, ball, cam)
_, _, stab_t = compute_possession(ps, bs, {1: "#1"}, DT, params)
print("touches raw:", len(raw_t), "stabilised:", len(stab_t))
assert len(raw_t) > 1 and len(stab_t) == 1, (len(raw_t), len(stab_t))

# ---- 5. pitch calibration on one frame, positions recovered at every other frame
# a 30 x 30 m rectangle on the ground; (u, v) are pitch metres
X0, Z0, L, Wp = -5.0, 20.0, 30.0, 30.0
c_frame = 45                                                     # calibrate mid-pan
corners = [(0, 0), (L, 0), (L, Wp), (0, Wp)]
img_pts = [project((X0 + u, CAM_H, Z0 + v), R_cam[c_frame]).tolist() for u, v in corners]
Hc = homography(img_pts, L, Wp)
players = {1: (5.0, 5.0), 2: (25.0, 20.0), 3: (15.0, 28.0)}
P3 = pd.concat([person((X0 + u, Z0 + v), tid) for tid, (u, v) in players.items()])
exact = CameraModel(table, W, H, f=F_TRUE)
for label, model, tol in (("true f", exact, 0.05), ("estimated f", cam, 0.6)):
    pst, _ = stabilise(P3, pd.DataFrame(columns=["frame", "t", "x1", "y1", "x2", "y2", "conf"]), model)
    pos = player_positions(pst, {1: "a", 2: "b", 3: "c"}, model.pitch_mapper(Hc, c_frame))
    worst = 0.0
    for tid, name in ((1, "a"), (2, "b"), (3, "c")):
        d = pos[pos["player"] == name]
        worst = max(worst, np.nanmax(np.hypot(d["X"] - players[tid][0], d["Y"] - players[tid][1])))
    print(f"pitch position error ({label}): worst {worst:.3f} m")
    assert worst < tol, (label, worst)
print("OK")
