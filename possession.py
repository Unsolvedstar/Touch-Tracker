"""Touches and time-on-ball from player + ball positions.

Pure numpy/pandas, no video or ML dependencies, so it can be tested on its own.

Definitions (shown in the app so the numbers can be interpreted):
  * A player is "in control" on a sampled frame when the ball is within
    `radius_factor` x (player box height) of that player's feet and they are
    the nearest player to it.
  * A "spell" is a run of in-control frames for one player. Short dropouts
    (ball briefly lost / occluded) are bridged, and spells shorter than
    `min_spell_s` are discarded as flicker.
  * Time on ball = total duration of that player's spells.
  * A touch = the start of a spell, plus every sharp change in ball velocity
    while the player is in control (a pass, shot, or each dribble contact).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OTHER = "_other"  # opposition, referees, unidentified: compete for the ball but aren't reported


@dataclass
class PossessionParams:
    radius_factor: float = 0.6       # control radius, in player-heights, from the feet (phone footage: ball position is approximate)
    min_spell_s: float = 0.3         # ignore control shorter than this
    bridge_gap_s: float = 0.5        # bridge ball dropouts / flicker up to this long
    max_ball_gap_s: float = 0.8      # interpolate missing ball detections up to this long
    max_ball_speed_h_s: float = 30.0  # player-heights/second; faster jumps are detection errors
    touch_dv_h_s: float = 2.0        # velocity change (player-heights/s) that counts as a touch
    touch_debounce_s: float = 0.25   # minimum time between two touches by the same spell
    hold_no_touch_s: float = 0.6     # a spell with no velocity change must last this long (else the ball just flew past)


def clean_ball(ball: pd.DataFrame, frames: np.ndarray, dt: float, vh: float, p: PossessionParams) -> pd.DataFrame:
    """One ball position per sampled frame (NaN where unknown): raw image coords (x, y) for proximity tests and
    camera-stabilised coords (sx, sy) for velocity. Rejects jumps, fills short gaps."""
    b = ball.sort_values("conf", ascending=False).drop_duplicates("frame").set_index("frame")
    b = b.reindex(frames)
    x = ((b["x1"] + b["x2"]) / 2).to_numpy(float)
    y = ((b["y1"] + b["y2"]) / 2).to_numpy(float)
    sx = b["sx"].to_numpy(float) if "sx" in b else x.copy()
    sy = b["sy"].to_numpy(float) if "sy" in b else y.copy()

    # Reject detections that imply an impossible speed relative to the last accepted one (judged against the
    # pitch, not the screen, so a panning camera doesn't cause false rejections).
    last = None
    for i in range(len(frames)):
        if np.isnan(x[i]):
            continue
        if last is not None:
            steps = max((i - last), 1)
            speed = np.hypot(sx[i] - sx[last], sy[i] - sy[last]) / (steps * dt) / vh
            if speed > p.max_ball_speed_h_s:
                x[i] = y[i] = sx[i] = sy[i] = np.nan
                continue
        last = i

    s = pd.DataFrame({"x": x, "y": y, "sx": sx, "sy": sy}, index=frames)
    limit = max(int(round(p.max_ball_gap_s / dt)), 1)
    return s.interpolate(limit=limit, limit_area="inside")


def _runs(ctrl: list) -> list[list]:
    """Run-length encode -> [[label, start_idx, end_idx_inclusive], ...]."""
    out: list[list] = []
    for i, c in enumerate(ctrl):
        if out and out[-1][0] == c:
            out[-1][2] = i
        else:
            out.append([c, i, i])
    return out


def compute_possession(
    players: pd.DataFrame,
    ball: pd.DataFrame,
    label_of: dict[int, str],
    dt: float,
    p: PossessionParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    players: frame, t, track_id, x1, y1, x2, y2   (all players, every team)
    ball:    frame, t, x1, y1, x2, y2, conf
    label_of: track_id -> player label. Tracks not in the map count as OTHER.
    dt: seconds between sampled frames.

    Returns (summary, events, touches):
      summary: one row per labelled player: touches, time_on_ball_s, spells, longest_spell_s
      events:  one row per spell: player, start_t, end_t, duration_s, touches, start_frame, end_frame
      touches: one row per touch: player, frame, t, x, y (ball in image pixels), sx, sy (camera-stabilised)
    """
    p = p or PossessionParams()
    empty_sum = pd.DataFrame(columns=["player", "touches", "time_on_ball_s", "spells", "longest_spell_s"])
    empty_ev = pd.DataFrame(columns=["player", "start_t", "end_t", "duration_s", "touches", "start_frame", "end_frame"])
    empty_tc = pd.DataFrame(columns=["player", "frame", "t", "x", "y", "sx", "sy"])
    if players.empty or ball.empty:
        return empty_sum, empty_ev, empty_tc

    frames = np.sort(players["frame"].unique())
    t_of = players.drop_duplicates("frame").set_index("frame")["t"]
    # vh: typical player height in the space velocities are measured in (stabilised when available)
    vh = float(players["sh"].median()) if "sh" in players else float((players["y2"] - players["y1"]).median())
    bpos = clean_ball(ball, frames, dt, vh, p)

    by_frame = {f: g for f, g in players.groupby("frame")}

    # 1) who is in control on each sampled frame
    ctrl: list = []
    for f in frames:
        bx, by = bpos.at[f, "x"], bpos.at[f, "y"]
        g = by_frame[f]
        if np.isnan(bx) or g.empty:
            ctrl.append(None)
            continue
        fx = ((g["x1"] + g["x2"]) / 2).to_numpy()
        fy = g["y2"].to_numpy()
        h = (g["y2"] - g["y1"]).to_numpy()
        d = np.hypot(fx - bx, fy - by)
        # ball above the feet (in the air / on the head) still counts: use the lower-body point too
        d = np.minimum(d, np.hypot(fx - bx, (fy - 0.25 * h) - by))
        ok = d <= p.radius_factor * h
        if not ok.any():
            ctrl.append(None)
            continue
        j = int(np.argmin(np.where(ok, d, np.inf)))
        tid = int(g["track_id"].iloc[j])
        ctrl.append(label_of.get(tid, OTHER))

    # 2) bridge short dropouts between two runs of the same controller, then drop flicker
    bridge_n = int(round(p.bridge_gap_s / dt))
    min_n = max(int(round(p.min_spell_s / dt)), 1)
    runs = _runs(ctrl)
    for k in range(1, len(runs) - 1):
        if runs[k][0] is None and (runs[k][2] - runs[k][1] + 1) <= bridge_n and runs[k - 1][0] == runs[k + 1][0]:
            runs[k][0] = runs[k - 1][0]
    ctrl2 = [None] * len(ctrl)
    for lab, s, e in _runs([r[0] for r in runs for _ in range(r[2] - r[1] + 1)]):
        if lab is not None and (e - s + 1) >= min_n:
            for i in range(s, e + 1):
                ctrl2[i] = lab
    spells = [r for r in _runs(ctrl2) if r[0] is not None]

    # 3) touches inside each spell = start + sharp ball-velocity changes
    bx = bpos["x"].to_numpy()
    by_ = bpos["y"].to_numpy()
    sbx = bpos["sx"].to_numpy()
    sby = bpos["sy"].to_numpy()
    vx = np.gradient(sbx) / dt / vh
    vy = np.gradient(sby) / dt / vh
    debounce_n = max(int(round(p.touch_debounce_s / dt)), 1)

    rows, trows = [], []
    for lab, s, e in spells:
        touch_idx = [s]
        for i in range(s + 1, e + 1):
            if np.isnan(vx[i]) or np.isnan(vx[i - 1]):
                continue
            dv = np.hypot(vx[i] - vx[i - 1], vy[i] - vy[i - 1])
            if dv >= p.touch_dv_h_s and i - touch_idx[-1] >= debounce_n:
                touch_idx.append(i)
        if len(touch_idx) == 1 and (e - s + 1) * dt < p.hold_no_touch_s:
            continue  # ball merely passed near the player: not a touch
        for ti in touch_idx:
            trows.append(dict(player=lab, frame=int(frames[ti]), t=float(t_of[frames[ti]]),
                              x=float(bx[ti]), y=float(by_[ti]), sx=float(sbx[ti]), sy=float(sby[ti])))
        rows.append(
            dict(
                player=lab,
                start_t=float(t_of[frames[s]]),
                end_t=float(t_of[frames[e]]) + dt,
                duration_s=(e - s + 1) * dt,
                touches=len(touch_idx),
                start_frame=int(frames[s]),
                end_frame=int(frames[e]),
            )
        )
    events = pd.DataFrame(rows, columns=empty_ev.columns)
    touches = pd.DataFrame(trows, columns=empty_tc.columns)
    if events.empty:
        return empty_sum, events, touches

    rep = events[events["player"] != OTHER]
    summary = (
        rep.groupby("player")
        .agg(
            touches=("touches", "sum"),
            time_on_ball_s=("duration_s", "sum"),
            spells=("touches", "size"),
            longest_spell_s=("duration_s", "max"),
        )
        .reset_index()
        .sort_values("touches", ascending=False)
        .reset_index(drop=True)
    )
    summary["time_on_ball_s"] = summary["time_on_ball_s"].round(2)
    summary["longest_spell_s"] = summary["longest_spell_s"].round(2)
    return summary, events, touches


def remove_static_balls(ball: pd.DataFrame, players: pd.DataFrame, min_span_s: float = 15.0,
                        max_spread_h: float = 0.35, min_hits: int = 15, min_duty: float = 0.25) -> tuple[pd.DataFrame, int]:
    """Drop "ball" detections that never move: spare balls on the touchline, a ball in a net, a white sock.
    Needs camera-stabilised positions (sx, sy) so a panning camera doesn't make them look like they move.
    A cluster of detections is static if it spans at least `min_span_s` seconds yet stays within
    `max_spread_h` player-heights, AND is present for at least `min_duty` of that span (a ball lying there the
    whole time). A match ball that returns to the same spot now and then (goal kicks) appears in short visits, so
    its duty is low and it is kept.
    Returns (filtered ball, number removed)."""
    if len(ball) < min_hits or "sx" not in ball:
        return ball, 0
    from sklearn.cluster import DBSCAN

    vh = float(players["sh"].median()) if "sh" in players else float((players["y2"] - players["y1"]).median())
    xy = ball[["sx", "sy"]].to_numpy()
    step = float(np.median(np.diff(np.sort(players["t"].unique())))) if players["t"].nunique() > 1 else 0.1
    labels = DBSCAN(eps=0.4 * vh, min_samples=min_hits).fit_predict(xy)
    drop = np.zeros(len(ball), bool)
    for lab in set(labels) - {-1}:
        m = labels == lab
        span = ball["t"].to_numpy()[m].max() - ball["t"].to_numpy()[m].min()
        spread = float(np.hypot(*xy[m].std(axis=0)))
        duty = m.sum() * step / max(span, step)
        if span >= min_span_s and spread <= max_spread_h * vh and duty >= min_duty:
            drop |= m
    return ball[~drop].reset_index(drop=True), int(drop.sum())


def remove_oversized_balls(ball: pd.DataFrame, players: pd.DataFrame, max_ratio: float = 0.22) -> tuple[pd.DataFrame, int]:
    """A match ball is about 1/8 of a player's height across. Detections much wider than that (relative to the
    tallest, i.e. nearest, players in the clip) are other things: balls lying beside the camera, a bag, a cone."""
    if ball.empty:
        return ball, 0
    ph = (players["y2"] - players["y1"]).quantile(0.9)
    w = np.maximum(ball["x2"] - ball["x1"], ball["y2"] - ball["y1"])
    keep = (w <= max_ratio * ph).to_numpy()
    return ball[keep].reset_index(drop=True), int((~keep).sum())


def filter_ball_candidates(ball: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Oversized and never-moving detections removed. Returns (ball, {'oversized': n, 'static': n})."""
    b, n_big = remove_oversized_balls(ball, players)
    b, n_static = remove_static_balls(b, players)
    return b, {"oversized": n_big, "static": n_static}
