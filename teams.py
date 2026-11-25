"""Group tracks into kit-colour clusters so the user can say which one is the home team."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cluster_tracks(players: pd.DataFrame, k: int = 4, min_samples: int = 10):
    """
    players needs columns track_id, r, g, b (torso colour per detection).
    Returns (track -> cluster id, cluster table with mean rgb + number of tracks).
    k=4 leaves room for home, away, goalkeepers and referees.
    """
    med = players.dropna(subset=["r", "g", "b"]).groupby("track_id").agg(
        r=("r", "median"), g=("g", "median"), b=("b", "median"), n=("r", "size")
    )
    med = med[med["n"] >= min_samples]
    if len(med) == 0:
        return {}, pd.DataFrame(columns=["cluster", "r", "g", "b", "tracks"])
    k = min(k, len(med))
    k = min(k, len(np.unique(med[["r", "g", "b"]].round(-1).to_numpy(), axis=0)))
    rgb = med[["r", "g", "b"]].to_numpy(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(lab)
    med["cluster"] = km.labels_
    table = (
        med.groupby("cluster")
        .agg(r=("r", "mean"), g=("g", "mean"), b=("b", "mean"), tracks=("n", "size"))
        .round()
        .astype(int)
        .reset_index()
    )
    return med["cluster"].to_dict(), table
