"""Run every check:  python run_tests.py   (add --video to also analyse a real clip from videos/; slow)."""
import os
import subprocess
import sys

TESTS = ["test_possession.py", "test_features.py", "test_camera.py", "test_ball_track.py", "test_identity.py",
         "test_library.py", "make_demo.py", "test_app.py", "test_grid.py"]
if "--video" in sys.argv:
    TESTS.append("test_upload_flow.py")

failed = []
for t in TESTS:
    r = subprocess.run([sys.executable, t], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(("ok    " if r.returncode == 0 else "FAIL  ") + t, flush=True)
    if r.returncode:
        failed.append(t)
        print((r.stdout + r.stderr)[-2000:])
print("all passed" if not failed else f"failed: {failed}")
sys.exit(bool(failed))
