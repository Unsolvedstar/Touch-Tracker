"""python run_ocr.py <run_name> ...  -> runs/<run>/numbers.csv (all tracks that have crops)."""
import sys, time
from pathlib import Path
import pandas as pd
from jersey import read_numbers

for name in sys.argv[1:]:
    d = Path("runs") / name
    ids = sorted({int(p.stem.split("_")[0]) for p in (d / "crops").glob("*.jpg")})
    t0 = time.time()
    last = [-1]
    def prog(f):
        if int(f * 10) != last[0]:
            last[0] = int(f * 10); print(f"{name} {f:.0%} {time.time()-t0:.0f}s", flush=True)
    res = read_numbers(d / "crops", ids, progress=prog)
    pd.DataFrame([(t, *v) for t, v in res.items()],
                 columns=["track_id", "number", "ocr_conf", "votes", "crops_read"]).to_csv(d / "numbers.csv", index=False)
    print(name, "done", sum(v[0] is not None for v in res.values()), "of", len(res), "tracks got a number", flush=True)
