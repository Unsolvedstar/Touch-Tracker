"""Command-line run of the detection step on a time range:  python run_clip.py VIDEO NAME START END [MODEL]"""
import sys, time
from pathlib import Path
import detect

video, name, start, end = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
model = sys.argv[5] if len(sys.argv) > 5 else "yolov8s.pt"
out = Path(__file__).parent / "runs" / name
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
last = [0]
def prog(f):
    if f - last[0] >= 0.1:
        last[0] = f; print(f"  {f:4.0%}  ({time.time() - t0:.0f}s)", flush=True)
meta = detect.run_detection(video, out, stride=None, model_name=model, imgsz=1280, progress=prog, start_s=start, end_s=end)
print("done in", round(time.time() - t0), "s; stride", meta["stride"], "dt", round(meta["dt"], 3), flush=True)
