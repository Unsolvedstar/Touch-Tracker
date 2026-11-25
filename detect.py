"""Video -> player tracks, ball detections, kit colours, shirt crops and camera motion
(YOLO + BoT-SORT + feature-based camera-motion estimation). Built for handheld / panning phone footage."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PERSON, BALL = 0, 32  # COCO class ids
CROPS_PER_TRACK = 6
GMC_WIDTH = 640           # camera-motion estimation runs on frames downscaled to this width
TARGET_ANALYSIS_HZ = 8    # auto stride aims for roughly this many analysed frames per second


def probe(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = dict(
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def auto_stride(fps: float) -> int:
    return max(1, int(round(fps / TARGET_ANALYSIS_HZ)))


def _torso(frame, b):
    """Median kit colour (RGB) of the upper body, ignoring grass-green pixels."""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    c = frame[max(int(y1 + 0.15 * h), 0): int(y1 + 0.5 * h), max(int(x1 + 0.25 * w), 0): int(x2 - 0.25 * w)]
    if c.size == 0:
        return (np.nan,) * 3
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    px = c.reshape(-1, 3)[~((hsv[:, 0] > 35) & (hsv[:, 0] < 85) & (hsv[:, 1] > 60))]
    if len(px) < 5:
        px = c.reshape(-1, 3)
    bgr = np.median(px, axis=0)
    return float(bgr[2]), float(bgr[1]), float(bgr[0])


def estimate_motion(prev_gray, gray, prev_boxes_small):
    """Homography mapping CURRENT-frame points to PREVIOUS-frame points (both downscaled coordinates),
    fitted on static background features (players are masked out). Returns (H or None, n_inliers)."""
    mask = np.full(prev_gray.shape, 255, np.uint8)
    for x1, y1, x2, y2 in prev_boxes_small:
        mask[max(int(y1), 0): int(y2) + 1, max(int(x1), 0): int(x2) + 1] = 0
    pts = cv2.goodFeaturesToTrack(prev_gray, 500, 0.01, 8, mask=mask)
    if pts is None or len(pts) < 20:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(31, 31), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None, int(ok.sum())
    H, inl = cv2.findHomography(nxt[ok], pts[ok], cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < 15:
        return None, 0 if inl is None else int(inl.sum())
    return H, int(inl.sum())


def run_detection(video_path: str, out_dir: Path, stride: int | None = None, model_name: str = "yolov8m.pt",
                  imgsz: int = 1280, progress=None, start_s: float = 0.0, end_s: float | None = None) -> dict:
    """Process every `stride`-th frame (None = auto from fps). Writes players.csv, ball.csv, camera.csv,
    crops/ and meta.json into out_dir."""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    info = probe(video_path)
    stride = stride or auto_stride(info["fps"])
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    # Optional time range. Start is snapped to a multiple of the stride so frame numbers stay on the stride grid.
    first = (int(start_s * info["fps"]) // stride) * stride
    last = info["frames"] if end_s is None else min(int(end_s * info["fps"]), info["frames"])
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    players, balls, cams = [], [], []
    best: dict[int, list] = {}  # track -> [(height, jpg bytes)]
    prev_gray, prev_boxes = None, []
    times = []
    i = first - 1
    while i + 1 < last and cap.grab():
        i += 1
        if i % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        H, W = frame.shape[:2]
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000.0 if (ms > 0 or i == 0) else i / info["fps"]  # real timestamps: phones record variable frame rate
        times.append(t)

        # camera motion relative to the previous analysed frame
        scale = min(1.0, GMC_WIDTH / W)
        small = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gmc_ok = 1
        step = np.eye(3)  # this frame -> previous analysed frame, in full-resolution pixels
        if prev_gray is not None:
            Hs, inliers = estimate_motion(prev_gray, gray, [[v * scale for v in b] for b in prev_boxes])
            if Hs is None:
                gmc_ok = 0  # couldn't estimate: assume the camera didn't move
            else:
                S = np.diag([scale, scale, 1.0])
                step = np.linalg.inv(S) @ Hs @ S
                step = step / step[2, 2]
        prev_gray = gray
        cams.append((i, *step.ravel().tolist(), gmc_ok))

        res = model.track(frame, persist=True, tracker="botsort.yaml", classes=[PERSON, BALL],
                          conf=0.1, imgsz=imgsz, verbose=False)[0]
        boxes = res.boxes
        prev_boxes = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(cls), -1)
            for b, c, cf, tid in zip(xyxy, cls, conf, ids):
                if c == BALL:
                    balls.append((i, t, *b, cf))
                elif c == PERSON and cf >= 0.3 and tid >= 0:
                    prev_boxes.append(b.tolist())
                    r, g, bl = _torso(frame, b)
                    players.append((i, t, int(tid), *b, cf, r, g, bl))
                    h = b[3] - b[1]
                    edge = b[0] < 2 or b[2] > W - 2 or b[1] < 2 or b[3] > H - 2
                    lst = best.setdefault(int(tid), [])
                    if not edge and (len(lst) < CROPS_PER_TRACK or h > min(x[0] for x in lst)):
                        x1, y1, x2, y2 = b
                        crop = frame[max(int(y1 + 0.1 * h), 0): int(y1 + 0.6 * h), max(int(x1), 0): int(x2)]
                        if crop.size:
                            lst.append((h, cv2.imencode(".jpg", crop)[1].tobytes()))
                            lst.sort(key=lambda x: -x[0])
                            del lst[CROPS_PER_TRACK:]
        if progress:
            progress(min((i - first) / max(last - first, 1), 1.0))
    cap.release()

    pcols = ["frame", "t", "track_id", "x1", "y1", "x2", "y2", "conf", "r", "g", "b"]
    bcols = ["frame", "t", "x1", "y1", "x2", "y2", "conf"]
    pd.DataFrame(players, columns=pcols).to_csv(out_dir / "players.csv", index=False)
    pd.DataFrame(balls, columns=bcols).to_csv(out_dir / "ball.csv", index=False)
    pd.DataFrame(cams, columns=["frame"] + [f"h{k}" for k in range(9)] + ["ok"]).to_csv(out_dir / "camera.csv", index=False)
    for tid, lst in best.items():
        for k, (_, jpg) in enumerate(lst):
            (out_dir / "crops" / f"{tid}_{k}.jpg").write_bytes(jpg)
    dts = np.diff(times)
    dt = float(np.median(dts[dts > 0])) if (dts > 0).any() else stride / info["fps"]
    meta = dict(info, stride=stride, dt=dt, model=model_name, video=str(video_path), frame_w=W, frame_h=H)
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def load_run(out_dir: Path):
    """-> (players, ball, meta, camera). players/ball carry stabilised coordinates (see camera.stabilise)."""
    from camera import CameraModel, load_camera, stabilise
    from library import find_video

    out_dir = Path(out_dir)
    players, ball = pd.read_csv(out_dir / "players.csv"), pd.read_csv(out_dir / "ball.csv")
    meta = json.loads((out_dir / "meta.json").read_text())
    meta["video"] = find_video(meta["video"], out_dir)  # in case it has been moved since the analysis
    table = load_camera(out_dir, players["frame"].unique())
    cam = CameraModel(table, meta.get("frame_w", meta["width"]), meta.get("frame_h", meta["height"]))
    players, ball = stabilise(players, ball, cam)
    cand_file = out_dir / "ball_candidates.csv"
    if cand_file.exists():  # deep ball search was run: pick the real ball by following candidates through time
        from ball_track import link_ball

        _, cands = stabilise(players.iloc[:0], pd.read_csv(cand_file), cam)
        ball = link_ball(cands, players)
    return players, ball, meta, cam
