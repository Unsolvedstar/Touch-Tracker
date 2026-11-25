import sys, time
from pathlib import Path
import ball
t0 = time.time(); last = [0]
def prog(f):
    if f - last[0] >= 0.1: last[0] = f; print(f"  {f:4.0%} ({time.time()-t0:.0f}s)", flush=True)
ball.run_ball_pass(Path(sys.argv[1]), progress=prog)
print("done in", round(time.time() - t0), "s", flush=True)
