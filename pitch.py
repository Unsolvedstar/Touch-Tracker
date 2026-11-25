"""Image -> pitch coordinates (metres) via a 4-point homography, plus heatmaps and movement stats.

Assumes a FIXED camera (tactical/wide cam). With a panning camera the calibration only holds
for the framing it was set on.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PRESETS = {
    "Full pitch (105 × 68 m)": (105.0, 68.0, True),
    "Penalty area (16.5 × 40.3 m)": (16.5, 40.32, False),
    "Custom rectangle": (30.0, 20.0, False),
}


def homography(img_pts, length: float, width: float) -> np.ndarray:
    """img_pts: 4 (x, y) pixel points in order top-left, top-right, bottom-right, bottom-left
    of the reference rectangle as seen in the image."""
    src = np.array(img_pts, np.float32)
    dst = np.array([[0, 0], [length, 0], [length, width], [0, width]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_pitch(H: np.ndarray, xs, ys) -> tuple[np.ndarray, np.ndarray]:
    pts = np.stack([np.asarray(xs, np.float32), np.asarray(ys, np.float32)], axis=1).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.array([]), np.array([])
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out[:, 0], out[:, 1]


def player_positions(players: pd.DataFrame, label_of: dict[int, str], H: np.ndarray) -> pd.DataFrame:
    """Foot position of every labelled player on every sampled frame, in metres."""
    d = players[players["track_id"].isin(label_of)].copy()
    d["player"] = d["track_id"].map(label_of)
    # H must take the same space as the points: stabilised when the run has camera data (see camera.py)
    fx, fy = (d["sfx"], d["sfy"]) if "sfx" in d else ((d["x1"] + d["x2"]) / 2, d["y2"])
    X, Y = H(fx, fy) if callable(H) else to_pitch(H, fx, fy)  # H: 3x3 matrix, or a mapper from CameraModel.pitch_mapper
    d["X"], d["Y"] = X, Y
    return d.groupby(["player", "frame"], as_index=False).agg(t=("t", "first"), X=("X", "mean"), Y=("Y", "mean"))


def movement_stats(pos: pd.DataFrame, max_speed_ms: float = 11.0) -> pd.DataFrame:
    """Distance and speed per player. Steps above max_speed_ms are treated as ID swaps/glitches and dropped."""
    rows = []
    for p, d in pos.groupby("player"):
        d = d.sort_values("t")
        x = d["X"].rolling(3, center=True, min_periods=1).median().to_numpy()
        y = d["Y"].rolling(3, center=True, min_periods=1).median().to_numpy()
        dt = np.diff(d["t"].to_numpy())
        step = np.hypot(np.diff(x), np.diff(y))
        ok = (dt > 0) & (dt <= 1.0)  # don't bridge long gaps in visibility
        speed = np.where(ok, step / np.where(dt > 0, dt, 1), np.nan)
        good = ok & (speed <= max_speed_ms)
        dist = float(step[good].sum())
        t_ok = float(dt[good].sum())
        rows.append(dict(player=p, distance_m=round(dist), minutes_tracked=round(t_ok / 60, 1),
                         avg_speed_kmh=round(3.6 * dist / t_ok, 1) if t_ok else None,
                         top_speed_kmh=round(3.6 * float(np.nanmax(np.where(good, speed, np.nan))), 1) if good.any() else None))
    return pd.DataFrame(rows).sort_values("distance_m", ascending=False).reset_index(drop=True)


def _pitch_shapes(L: float, W: float, full: bool):
    ln = dict(color="rgba(255,255,255,0.85)", width=1.5)
    s = [dict(type="rect", x0=0, y0=0, x1=L, y1=W, line=ln)]
    if full:
        s += [dict(type="line", x0=L / 2, y0=0, x1=L / 2, y1=W, line=ln),
              dict(type="circle", x0=L / 2 - 9.15, y0=W / 2 - 9.15, x1=L / 2 + 9.15, y1=W / 2 + 9.15, line=ln)]
        for x0, x1 in ((0, 16.5), (L - 16.5, L)):
            s.append(dict(type="rect", x0=x0, x1=x1, y0=W / 2 - 20.16, y1=W / 2 + 20.16, line=ln))
        for x0, x1 in ((0, 5.5), (L - 5.5, L)):
            s.append(dict(type="rect", x0=x0, x1=x1, y0=W / 2 - 9.16, y1=W / 2 + 9.16, line=ln))
    return s


def _base(L, W, full, title):
    fig = go.Figure()
    fig.update_layout(
        title=title, shapes=_pitch_shapes(L, W, full), plot_bgcolor="#2e7d32", height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=[-2, L + 2], showgrid=False, zeroline=False, visible=False, constrain="domain"),
        yaxis=dict(range=[W + 2, -2], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
    )
    return fig


def heatmap_figure(X, Y, L, W, full, title, bins_m: float = 3.0) -> go.Figure:
    fig = _base(L, W, full, title)
    X, Y = np.asarray(X), np.asarray(Y)
    keep = (X >= -5) & (X <= L + 5) & (Y >= -5) & (Y <= W + 5)
    if keep.sum() >= 3:
        nx, ny = max(int(L / bins_m), 4), max(int(W / bins_m), 4)
        h, xe, ye = np.histogram2d(X[keep], Y[keep], bins=[nx, ny], range=[[0, L], [0, W]])
        h = cv2.GaussianBlur(h.T.astype(np.float32), (0, 0), 1.0)
        h[h < h.max() * 0.04] = np.nan
        fig.add_trace(go.Heatmap(z=h, x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, colorscale="YlOrRd",
                                 opacity=0.75, showscale=False, hoverinfo="skip"))
    return fig


def touches_figure(X, Y, L, W, full, title) -> go.Figure:
    fig = _base(L, W, full, title)
    fig.add_trace(go.Scatter(x=X, y=Y, mode="markers", marker=dict(size=9, color="white", line=dict(color="black", width=1)),
                             hovertemplate="x %{x:.0f} m, y %{y:.0f} m<extra></extra>"))
    return fig


def network_figure(avg_pos: dict, matrix: pd.DataFrame, L, W, full, title="Pass network") -> go.Figure:
    """Players at their average position; line width = completed passes between the pair (both directions)."""
    fig = _base(L, W, full, title)
    names = [p for p in matrix.index if p in avg_pos]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = int(matrix.loc[a, b] + matrix.loc[b, a])
            if n:
                fig.add_trace(go.Scatter(x=[avg_pos[a][0], avg_pos[b][0]], y=[avg_pos[a][1], avg_pos[b][1]], mode="lines",
                                         line=dict(color="white", width=1 + 2 * n), opacity=0.6, hoverinfo="skip",
                                         showlegend=False))
    fig.add_trace(go.Scatter(x=[avg_pos[p][0] for p in names], y=[avg_pos[p][1] for p in names], mode="markers+text",
                             text=names, textposition="top center", textfont=dict(color="white"),
                             marker=dict(size=14, color="#ffeb3b", line=dict(color="black", width=1)), showlegend=False,
                             hoverinfo="text"))
    return fig
