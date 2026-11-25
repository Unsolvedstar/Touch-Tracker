"""Camera-motion compensation for handheld / panning phone footage.

A phone filming from the touchline mostly ROTATES (pan/tilt) about a fixed spot. For a rotating camera the
image motion between two frames is a homography H = K R K^-1, so from the per-step homographies that
detect.py stores we can recover the camera rotation R for every analysed frame (relative to the first one),
once the focal length f is known (estimated from the footage itself).

Rotations don't blow up the way chained planar matrices do over a 90-degree swing, so everything is measured
in camera-independent angular ("cylindrical") coordinates:

    ray = R_t K^-1 [x, y, 1]            direction of a pixel in the reference frame
    sx  = f * atan2(rx, rz)             "pixels" along the horizon
    sy  = f * ry / sqrt(rx^2 + rz^2)    "pixels" up/down

Columns added by stabilise():
  players: sfx, sfy (feet)   sh (player height in these units)
  ball:    sx, sy (centre)

Pitch positions: re-project a stabilised point into the camera view of a calibration frame, then apply the
4-point homography the user set up on that frame (see CameraModel.pitch_mapper).

Limits: assumes no zoom change during the clip, and that the phone rotates rather than travels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HCOLS = [f"h{k}" for k in range(9)]
MCOLS = [f"m{k}" for k in range(9)]


def _unit_homography(H: np.ndarray) -> np.ndarray:
    return H / np.cbrt(np.linalg.det(H)) if abs(np.linalg.det(H)) > 1e-12 else np.eye(3)


def load_camera(out_dir: Path, frames) -> pd.DataFrame:
    """Per-frame STEP homographies (current frame -> previous analysed frame, full-res pixels): frame, h0..h8, ok.
    Older runs stored cumulative matrices (m0..m8): converted. Runs with no camera data get a steady camera."""
    f = Path(out_dir) / "camera.csv"
    if not f.exists():
        return pd.DataFrame([[int(fr), *np.eye(3).ravel(), 1] for fr in sorted(frames)], columns=["frame"] + HCOLS + ["ok"])
    c = pd.read_csv(f).sort_values("frame").reset_index(drop=True)
    if "h0" in c:
        return c
    M = c[MCOLS].to_numpy(float).reshape(-1, 3, 3)
    steps = [np.eye(3)] + [np.linalg.inv(M[i - 1]) @ M[i] for i in range(1, len(M))]
    steps = [s / s[2, 2] for s in steps]
    out = pd.DataFrame(np.array(steps).reshape(-1, 9), columns=HCOLS)
    out.insert(0, "frame", c["frame"].to_numpy())
    out["ok"] = c["ok"].to_numpy()
    return out


def _K(f: float, W: float, H: float) -> np.ndarray:
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def _nearest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else U @ np.diag([1, 1, -1]) @ Vt


def estimate_focal(steps: np.ndarray, ok: np.ndarray, W: float, H: float) -> tuple[float, bool]:
    """Focal length (pixels) for which K^-1 H K is closest to a rotation, over steps with visible motion.
    Returns (f, estimated). Falls back to a typical phone main camera (~65 deg field of view) if too little motion."""
    default = 0.8 * W
    shift = np.abs(steps[:, 0, 2]) + np.abs(steps[:, 1, 2])
    use = np.where(ok.astype(bool) & (shift > 0.01 * W))[0]
    if len(use) < 8:
        return default, False
    use = use[:: max(1, len(use) // 200)]
    cands = np.geomspace(0.4 * W, 3.0 * W, 80)
    scores = []
    for f in cands:
        K, Ki = _K(f, W, H), np.linalg.inv(_K(f, W, H))
        e = []
        for i in use:
            A = _unit_homography(Ki @ steps[i] @ K)
            e.append(np.linalg.norm(A @ A.T - np.eye(3)))
        scores.append(np.median(e))
    return float(cands[int(np.argmin(scores))]), True


class CameraModel:
    def __init__(self, table: pd.DataFrame, W: float, H: float, f: float | None = None):
        self.table, self.W, self.H = table, float(W), float(H)
        self.frames = table["frame"].to_numpy()
        steps = table[HCOLS].to_numpy(float).reshape(-1, 3, 3)
        ok = table["ok"].to_numpy()
        if f is None:
            f, self.f_estimated = estimate_focal(steps, ok, W, H)
        else:
            self.f_estimated = True
        self.f = float(f)
        K, Ki = _K(self.f, W, H), np.linalg.inv(_K(self.f, W, H))
        R = np.eye(3)
        Rs = []
        for i, S in enumerate(steps):
            if i > 0:  # ray_prev = Rstep * ray_cur
                R = R @ _nearest_rotation(_unit_homography(Ki @ S @ K))
            Rs.append(R)
        self.R = np.array(Rs)

    @property
    def ok_fraction(self) -> float:
        return float((self.table["ok"] == 1).mean())

    def _R_for(self, frames) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.frames, np.asarray(frames), side="right") - 1, 0, len(self.R) - 1)
        return self.R[idx]

    def to_cyl(self, frames, x, y):
        """Pixel positions seen at `frames` -> camera-stabilised cylindrical coordinates."""
        R = self._R_for(frames)
        ray = np.einsum("nij,nj->ni", R, np.stack([(np.asarray(x, float) - self.W / 2) / self.f,
                                                   (np.asarray(y, float) - self.H / 2) / self.f,
                                                   np.ones(len(R))], axis=1))
        rxz = np.hypot(ray[:, 0], ray[:, 2])
        return self.f * np.arctan2(ray[:, 0], ray[:, 2]), self.f * ray[:, 1] / rxz

    def pan_degrees(self) -> float:
        """Largest angle the optical axis ever swings away from where the first frame pointed."""
        axes = self.R[:, :, 2]
        return float(np.degrees(np.arccos(np.clip(axes @ axes[0], -1, 1))).max())

    def pitch_mapper(self, H_img_to_pitch: np.ndarray, calib_frame: int):
        """-> fn(sx, sy) -> (X, Y) metres. H_img_to_pitch is a homography set up on `calib_frame`'s pixels.
        Valid for points the calibration camera could have seen (in front of it)."""
        Rc = self._R_for([calib_frame])[0]

        def fn(sx, sy):
            sx, sy = np.asarray(sx, float), np.asarray(sy, float)
            if sx.size == 0:
                return np.array([]), np.array([])
            th = sx / self.f
            ray = np.stack([np.sin(th), sy / self.f, np.cos(th)], axis=1)      # reference-frame direction
            rc = ray @ Rc                                                       # = Rc^T ray, in calibration camera
            z = np.where(rc[:, 2] > 0.05, rc[:, 2], np.nan)
            px, py = self.f * rc[:, 0] / z + self.W / 2, self.f * rc[:, 1] / z + self.H / 2
            from pitch import to_pitch

            return to_pitch(H_img_to_pitch, px, py)

        return fn


def stabilise(players: pd.DataFrame, ball: pd.DataFrame, cam: CameraModel):
    players, ball = players.copy(), ball.copy()
    if len(players):
        cx = ((players["x1"] + players["x2"]) / 2).to_numpy(float)
        fx, fy = cam.to_cyl(players["frame"].to_numpy(), cx, players["y2"].to_numpy(float))
        hx, hy = cam.to_cyl(players["frame"].to_numpy(), cx, players["y1"].to_numpy(float))
        players["sfx"], players["sfy"], players["sh"] = fx, fy, np.hypot(fx - hx, fy - hy)
    else:
        for c in ("sfx", "sfy", "sh"):
            players[c] = []
    if len(ball):
        bx, by = cam.to_cyl(ball["frame"].to_numpy(), ((ball["x1"] + ball["x2"]) / 2).to_numpy(float),
                            ((ball["y1"] + ball["y2"]) / 2).to_numpy(float))
        ball["sx"], ball["sy"] = bx, by
    else:
        ball["sx"], ball["sy"] = [], []
    return players, ball
