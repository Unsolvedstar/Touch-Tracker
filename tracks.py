"""Stitch fragmented tracker IDs: one player who lost their ID (occlusion, tackle, edge of frame)
and was re-detected under a new one."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stitch_tracks(players: pd.DataFrame, track_ids, dt: float,
                  max_gap_s: float = 6.0, max_speed_h_s: float = 3.0,
                  max_size_ratio: float = 1.6, max_colour_dist: float = 0.35,
                  numbers: dict[int, int] | None = None) -> dict[int, int]:
    """
    Link track A -> track B when B starts shortly after A ends, close to where A was last seen
    (allowing a plausible running speed, in player-heights per second).
    Only tracks in `track_ids` are considered, so pass one team's tracks at a time.
    `numbers` ({track_id: shirt number}, only trustworthy reads) adds two rules: two tracks with different
    numbers are never joined, and chains that carry the same number are joined however long the gap, provided
    they were never on screen at the same time (one player can't be in two places at once).
    Returns {track_id: root_track_id} for every id in track_ids (root = earliest track of the chain).
    """
    numbers = {int(k): int(v) for k, v in (numbers or {}).items() if int(k) in set(int(t) for t in track_ids)}
    ids = set(int(t) for t in track_ids)
    g = players[players["track_id"].isin(ids)]
    fx, fy, hh = ("sfx", "sfy", "sh") if "sfx" in g else (None, None, None)  # stabilised if available
    info = {}
    for tid, d in g.groupby("track_id"):
        d = d.sort_values("frame")
        a, b = d.iloc[0], d.iloc[-1]
        rgb = d[["r", "g", "b"]].median().to_numpy(float) if {"r", "g", "b"} <= set(d.columns) else None
        info[int(tid)] = dict(
            rgb=rgb, n=len(d),
            f0=a["frame"], f1=b["frame"], t0=a["t"], t1=b["t"],
            p0=(a[fx], a[fy]) if fx else ((a["x1"] + a["x2"]) / 2, a["y2"]),
            p1=(b[fx], b[fy]) if fx else ((b["x1"] + b["x2"]) / 2, b["y2"]),
            h=float(d[hh].median()) if hh else float((d["y2"] - d["y1"]).median()),
        )

    cands = []
    for a, ia in info.items():
        for b, ib in info.items():
            if a == b or ib["f0"] <= ia["f1"]:
                continue
            gap = ib["t0"] - ia["t1"]
            if gap > max_gap_s:
                continue
            h = (ia["h"] + ib["h"]) / 2
            dist = float(np.hypot(ib["p0"][0] - ia["p1"][0], ib["p0"][1] - ia["p1"][1]))
            allowed = max_speed_h_s * h * max(gap, dt) + 0.3 * h
            if dist > allowed:
                continue
            # Same player should look the same: similar apparent size and similar shirt colour.
            size_ratio = max(ia["h"], ib["h"]) / max(min(ia["h"], ib["h"]), 1e-6)
            if size_ratio > max_size_ratio:
                continue
            app = 0.0
            if ia["rgb"] is not None and ib["rgb"] is not None:
                app = float(np.linalg.norm(ia["rgb"] - ib["rgb"])) / 255.0
                if app > max_colour_dist:
                    continue
            # Cost: how far (vs how far they could have run), how long the gap, how different they look.
            cands.append((dist / allowed + 0.05 * gap + 2.0 * app, a, b))

    succ: dict[int, int] = {}
    pred: dict[int, int] = {}

    def chain_numbers(t: int) -> set:
        nums, u = set(), t
        while u in pred:
            u = pred[u]
        while True:
            if u in numbers:
                nums.add(numbers[u])
            if u not in succ:
                return nums
            u = succ[u]

    for _, a, b in sorted(cands):
        if a in succ or b in pred:
            continue
        if len(chain_numbers(a) | chain_numbers(b)) > 1:  # two different shirt numbers: different players
            continue
        succ[a], pred[b] = b, a

    root = {}
    for t in info:
        r = t
        while r in pred:
            r = pred[r]
        root[t] = r

    # Same shirt number, any gap: join the chains unless they overlap in time.
    if numbers:
        frames_of = {t: set(d["frame"]) for t, d in g.groupby("track_id")}
        chains: dict[int, list[int]] = {}
        for t, r in root.items():
            chains.setdefault(r, []).append(t)
        by_num: dict[int, list[int]] = {}
        for r, mem in chains.items():
            nums = {numbers[t] for t in mem if t in numbers}
            if len(nums) == 1:
                by_num.setdefault(nums.pop(), []).append(r)
        for n, rs in by_num.items():
            rs.sort(key=lambda r: info[r]["f0"])
            groups: list[tuple[int, set]] = []  # (root, frames used)
            for r in rs:
                fr = set().union(*(frames_of[t] for t in chains[r]))
                for gi, (gr, gf) in enumerate(groups):
                    if not (fr & gf):
                        for t in chains[r]:
                            root[t] = gr
                        groups[gi] = (gr, gf | fr)
                        break
                else:
                    groups.append((r, fr))
    for t in ids - set(root):  # tracks with no rows (shouldn't happen) map to themselves
        root[t] = t
    return root
