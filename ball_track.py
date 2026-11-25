"""Decide which ball candidates are the ball by following them through time.

A real ball moves along a smooth, physically possible path; false positives (a white sock, a patch of
grass) appear and vanish. Candidates are linked across analysed frames in camera-STABILISED coordinates, so a
panning phone doesn't break the physics, and only chains of consistent candidates are kept.

Dynamic programming over candidates sorted by time:
  score(j) = unary(j) + max(0, best over earlier i within `max_gap_s` of  score(i) + transition(i, j))
  unary      = confidence (log-odds) + bonus if near a player, penalty if far from every player
  transition = - speed penalty - per-skipped-frame penalty;  forbidden above `max_speed_h_s` (player-heights / s)
Chains are then read off greedily, best first, and kept only if long enough and confident enough. At most one
ball per frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LinkParams:
    max_gap_s: float = 1.0           # a ball can go unseen this long inside a chain
    max_speed_h_s: float = 20.0      # player-heights per second (~36 m/s): faster = not the same ball
    speed_cost: float = 0.15         # per (player-height / s)
    skip_cost: float = 0.25          # per analysed frame skipped inside a chain
    near_player_h: float = 2.5       # "near play" radius in player-heights
    near_bonus: float = 0.6
    unary_offset: float = 1.5        # shifts detection evidence so a modest-confidence ball can pay for its links
    far_penalty: float = 1.0
    min_chain: int = 5               # detections needed for a chain to count
    min_mean_unary: float = 0.0      # average per-detection evidence needed (log-odds scale)


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def link_ball(cands: pd.DataFrame, players: pd.DataFrame, params: LinkParams | None = None) -> pd.DataFrame:
    """cands: frame, t, x1, y1, x2, y2, conf, sx, sy (stabilised centre).  players: frame, sfx, sfy, sh.
    Returns the chosen candidates (subset of rows, <= 1 per frame)."""
    p = params or LinkParams()
    if cands.empty:
        return cands
    c = cands.sort_values(["t", "conf"], ascending=[True, False]).reset_index(drop=True)
    n = len(c)
    t, x, y, frame = c["t"].to_numpy(float), c["sx"].to_numpy(float), c["sy"].to_numpy(float), c["frame"].to_numpy()
    vh = float(players["sh"].median())
    step = float(np.median(np.diff(np.sort(players["t"].unique())))) if players["t"].nunique() > 1 else 0.1

    # unary evidence
    u = _logit(c["conf"].to_numpy(float)) + p.unary_offset
    pl = {f: g[["sfx", "sfy"]].to_numpy() for f, g in players.groupby("frame")}
    for i in range(n):
        g = pl.get(frame[i])
        if g is None or not len(g):
            continue
        d = np.hypot(g[:, 0] - x[i], g[:, 1] - y[i]).min() / vh
        u[i] += p.near_bonus if d <= p.near_player_h else -p.far_penalty

    score = u.copy()
    prev = np.full(n, -1)
    lo = 0
    for j in range(n):
        while t[j] - t[lo] > p.max_gap_s:
            lo += 1
        best, arg = 0.0, -1
        for i in range(lo, j):
            dt = t[j] - t[i]
            if dt <= 1e-9:                     # same frame: alternatives, not a sequence
                continue
            speed = np.hypot(x[j] - x[i], y[j] - y[i]) / dt / vh
            if speed > p.max_speed_h_s:
                continue
            skipped = max(int(round(dt / step)) - 1, 0)
            s = score[i] - p.speed_cost * speed - p.skip_cost * skipped
            if s > best:
                best, arg = s, i
        if arg >= 0:
            score[j] = u[j] + best
            prev[j] = arg

    # read chains off, best score first; one ball per frame
    used = np.zeros(n, bool)
    claimed_frames: set = set()
    chosen: list[int] = []
    for j in np.argsort(-score):
        if used[j]:
            continue
        chain, k = [], j
        while k >= 0 and not used[k]:
            chain.append(k)
            k = prev[k]
        chain = [k for k in chain if frame[k] not in claimed_frames]
        if len(chain) >= p.min_chain and float(np.mean(u[chain])) >= p.min_mean_unary:
            chosen += chain
            claimed_frames |= {frame[k] for k in chain}
        used[chain] = True
    return c.loc[sorted(chosen)].reset_index(drop=True)
