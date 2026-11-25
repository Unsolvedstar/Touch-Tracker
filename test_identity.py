import tempfile
from pathlib import Path

import pandas as pd

from identity import NOT_OURS, assign, load_names, merge_keys, on_screen_twice, root_names, save_names
from tracks import stitch_tracks

rows = []
def add(tid, t0, t1, x):
    for t in range(t0, t1):
        rows.append(dict(frame=t, t=float(t), track_id=tid, x1=x, x2=x + 20, y1=0, y2=60, r=200, g=30, b=30))
add(1, 0, 5, 100); add(2, 30, 40, 900); add(3, 30, 40, 500); add(4, 41, 50, 520); add(5, 60, 70, 100)
P = pd.DataFrame(rows)

def stitch(names, numbers=None):
    return stitch_tracks(P, [1, 2, 3, 4, 5], 1.0, numbers=merge_keys(names, numbers or {}))

# saved names round-trip (including non-ASCII) and can be cleared
with tempfile.TemporaryDirectory() as d:
    n = assign({}, [1, 2], " Thandō ")
    save_names(Path(d), n)
    assert load_names(Path(d)) == {1: "Thandō", 2: "Thandō"}
    assert assign(n, [1], "") == {2: "Thandō"}

# same name joins across any gap; position-chained tracks inherit it
r = stitch({1: "Sipho", 2: "Sipho"})
assert r[1] == r[2], r
assert r[3] == r[4], r
members = {}
for t, root in r.items():
    members.setdefault(root, []).append(t)
assert root_names(members, {3: "Ayo"})[r[4]] == "Ayo"

# different names are never joined, even when position would join them
assert stitch({3: "Ayo", 4: "Ben"})[4] != 3

# a shirt number seen on a named track pulls other tracks with that number to the name
r = stitch({1: "Sipho"}, {1: 10, 5: 10})
assert r[5] == r[1], r

# co-visible tracks with one name stay apart and are reported; "not ours" is never reported
r = stitch({2: "Sipho", 3: "Sipho"})
assert r[2] != r[3]
assert on_screen_twice(P, {2: "Sipho", 3: "Sipho"}) == {"Sipho": 10}
assert on_screen_twice(P, {2: NOT_OURS, 3: NOT_OURS}) == {}
print("OK")
