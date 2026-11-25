"""Passes, receptions, turnovers from the ordered list of possession spells.

Rule: two consecutive spells, with at most `max_gap_s` between the first ending and the second
starting, form a ball transfer.
  home A -> home B  (A != B)  : completed pass A -> B
  home A -> other             : ball lost by A (intercepted / tackled / lost)
  other  -> home B            : ball won by B
Gaps longer than max_gap_s (ball out of play, long clearance, lost detection) are not counted.
Shots and clearances that go out of play are therefore not counted as passes.
"""
from __future__ import annotations

import pandas as pd

from possession import OTHER


def detect_passes(events: pd.DataFrame, max_gap_s: float = 3.0) -> pd.DataFrame:
    cols = ["t", "from", "to", "outcome", "gap_s"]
    if len(events) < 2:
        return pd.DataFrame(columns=cols)
    ev = events.sort_values("start_t").reset_index(drop=True)
    rows = []
    for i in range(len(ev) - 1):
        a, b = ev.iloc[i], ev.iloc[i + 1]
        gap = b["start_t"] - a["end_t"]
        if gap > max_gap_s or a["player"] == b["player"]:
            continue
        a_home, b_home = a["player"] != OTHER, b["player"] != OTHER
        if a_home and b_home:
            outcome = "completed"
        elif a_home:
            outcome = "lost"
        elif b_home:
            outcome = "won"
        else:
            continue
        rows.append(dict(t=a["end_t"], **{"from": a["player"], "to": b["player"], "outcome": outcome,
                                          "gap_s": round(max(gap, 0.0), 2)}))
    return pd.DataFrame(rows, columns=cols)


def pass_summary(passes: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    rows = []
    for p in players:
        made = passes[(passes["from"] == p) & (passes["outcome"] == "completed")]
        lost = passes[(passes["from"] == p) & (passes["outcome"] == "lost")]
        recv = passes[(passes["to"] == p) & (passes["outcome"] == "completed")]
        won = passes[(passes["to"] == p) & (passes["outcome"] == "won")]
        n_out = len(made) + len(lost)
        rows.append(dict(player=p, passes_completed=len(made), ball_lost=len(lost),
                         pass_completion_pct=round(100 * len(made) / n_out, 0) if n_out else None,
                         received=len(recv), ball_won=len(won)))
    return pd.DataFrame(rows).sort_values("passes_completed", ascending=False).reset_index(drop=True)


def pass_matrix(passes: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    done = passes[passes["outcome"] == "completed"]
    m = pd.crosstab(done["from"], done["to"]).reindex(index=players, columns=players, fill_value=0)
    m.index.name, m.columns.name = "from", "to"
    return m
