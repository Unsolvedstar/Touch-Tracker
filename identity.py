"""Names the coach assigns by tapping photos.

Stored per RAW tracker id in runs/<run>/names.json, so they survive changes to the merge settings.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

NOT_OURS = "__not_ours__"  # opponent, referee, spectator: leave out of the home stats


def load_names(run_dir: Path) -> dict[int, str]:
    f = Path(run_dir) / "names.json"
    if not f.exists():
        return {}
    return {int(k): v for k, v in json.loads(f.read_text(encoding="utf8")).items()}


def save_names(run_dir: Path, names: dict[int, str]) -> None:
    f = Path(run_dir) / "names.json"
    f.write_text(json.dumps({str(k): v for k, v in sorted(names.items())}, ensure_ascii=False, indent=1), encoding="utf8")


def assign(names: dict[int, str], track_ids, name: str | None) -> dict[int, str]:
    """-> a copy with `name` set on (or, if None/blank, removed from) every id in track_ids."""
    out = dict(names)
    name = (name or "").strip()
    for t in track_ids:
        if name:
            out[int(t)] = name
        else:
            out.pop(int(t), None)
    return out


def merge_keys(names: dict[int, str], numbers: dict[int, int]) -> dict[int, int]:
    """Combine names and trusted shirt numbers into the `numbers` argument of tracks.stitch_tracks:
    same key = same player, different keys = never join. Names win over numbers. A number that has been seen on
    exactly one name means that name for other tracks too."""
    ids = {n: -(i + 1) for i, n in enumerate(sorted(set(names.values())))}  # negative: can't clash with shirt numbers
    names_of_number: dict[int, set] = {}
    for t, num in numbers.items():
        if t in names and names[t] != NOT_OURS:
            names_of_number.setdefault(num, set()).add(names[t])
    keys = {}
    for t, num in numbers.items():
        owners = names_of_number.get(num, set())
        keys[t] = ids[next(iter(owners))] if len(owners) == 1 else num
    keys.update({t: ids[n] for t, n in names.items()})
    return keys


def root_names(members: dict[int, list[int]], names: dict[int, str]) -> dict[int, str]:
    """Name of each merged player: the most common name among its raw tracks (unnamed ones inherit it)."""
    out = {}
    for root, ids in members.items():
        found = Counter(names[t] for t in ids if t in names)
        if found:
            out[root] = found.most_common(1)[0][0]
    return out


def on_screen_twice(players: pd.DataFrame, names: dict[int, str]) -> dict[str, int]:
    """{name: number of analysed frames in which two tracks with that name are visible at once}.
    One player can't be in two places, so each is a naming mistake (or a duplicate detection)."""
    d = players[["frame", "track_id"]].copy()
    d["name"] = d["track_id"].map(names)
    d = d[d["name"].notna() & (d["name"] != NOT_OURS)]
    per = d.groupby(["name", "frame"])["track_id"].nunique()
    bad = per[per > 1]
    return {} if bad.empty else bad.groupby(level="name").size().to_dict()
