"""Render an annotated copy of the video: boxes, names, who is on the ball, live touch counters."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GREEN, YELLOW, GREY, RED = (60, 200, 60), (0, 230, 255), (170, 170, 170), (0, 0, 255)


def _h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so browsers can play it. Returns False if no ffmpeg is available."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe:
        return False
    r = subprocess.run([exe, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                        "-crf", "23", str(dst)], capture_output=True)
    return r.returncode == 0 and dst.exists()


def render_annotated(meta: dict, players: pd.DataFrame, ball: pd.DataFrame, label_of: dict[int, str],
                     events: pd.DataFrame, touches: pd.DataFrame, out_path: Path,
                     start_s: float = 0.0, end_s: float = 60.0, progress=None) -> tuple[Path, bool]:
    """Returns (path, browser_playable)."""
    out_path = Path(out_path)
    fps, stride = meta["fps"], meta["stride"]
    cap = cv2.VideoCapture(meta["video"])
    f0 = int(start_s * fps)
    f1 = min(int(end_s * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, first = cap.read()  # take the output size from a real frame (phone videos can be rotated by metadata)
    if not ok:
        raise RuntimeError("Could not read the video")
    H, W = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    # per analysed frame: {track_id: box} and ball centre, so boxes can be interpolated between analysed frames
    by_frame = {f: {int(r.track_id): (r.x1, r.y1, r.x2, r.y2) for r in g.itertuples()} for f, g in players.groupby("frame")}
    ball_by = {f: ((g.x1.iloc[0] + g.x2.iloc[0]) / 2, (g.y1.iloc[0] + g.y2.iloc[0]) / 2) for f, g in
               ball.sort_values("conf", ascending=False).drop_duplicates("frame").groupby("frame")}
    ev = events.sort_values("start_t")
    home_touch = touches[touches["player"].isin(set(label_of.values()))]
    tt = {p: np.sort(g["t"].to_numpy()) for p, g in home_touch.groupby("player")}
    order = sorted(tt, key=lambda p: -len(tt[p]))[:12]

    tmp = out_path.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = max(f1 - f0, 1)
    for k, f in enumerate(range(f0, f1)):
        ok, img = cap.read()
        if not ok:
            break
        t = f / fps
        sf = (f // stride) * stride  # nearest analysed frame at or before this one
        active = ev[(ev["start_t"] <= t) & (ev["end_t"] >= t)]
        holder = active["player"].iloc[0] if len(active) else None
        alpha = (f - sf) / stride
        cur, nxt = by_frame.get(sf, {}), by_frame.get(sf + stride, {})
        for tid, bx in cur.items():
            if tid in nxt:
                bx = tuple((1 - alpha) * a + alpha * b for a, b in zip(bx, nxt[tid]))
            elif alpha >= 0.5:
                continue  # track ends: stop drawing it halfway to the next analysed frame
            lab = label_of.get(tid)
            p1, p2 = (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3]))
            if lab is None:
                cv2.rectangle(img, p1, p2, GREY, 1)
                continue
            on = lab == holder
            cv2.rectangle(img, p1, p2, YELLOW if on else GREEN, 3 if on else 2)
            cv2.putText(img, lab, (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, YELLOW if on else GREEN, 2, cv2.LINE_AA)
        b0, b1 = ball_by.get(sf), ball_by.get(sf + stride)
        if b0 and b1:
            b0 = ((1 - alpha) * b0[0] + alpha * b1[0], (1 - alpha) * b0[1] + alpha * b1[1])
        if b0 and (b1 or alpha < 0.5):
            cv2.circle(img, (int(b0[0]), int(b0[1])), 10, RED, 2)
        if order:  # live touch counter
            ov = img.copy()
            cv2.rectangle(ov, (10, 10), (290, 20 + 24 * len(order)), (0, 0, 0), -1)
            img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
            for i, p in enumerate(order):
                c = int(np.searchsorted(tt[p], t, side="right"))
                cv2.putText(img, f"{p[:18]:<18} {c:>3}", (18, 34 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            YELLOW if p == holder else (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        if progress and k % 10 == 0:
            progress(k / n)
    vw.release()
    cap.release()

    playable = _h264(tmp, out_path)
    if playable:
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(out_path)
    return out_path, playable
