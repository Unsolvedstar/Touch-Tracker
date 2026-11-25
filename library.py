"""Where videos and analyses live on this computer."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "videos"
VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mkv")


def local_videos() -> list[Path]:
    """Videos dropped into videos/ (or next to the app): used directly, no upload or copy."""
    found = [p for d in (VIDEO_DIR, ROOT) if d.is_dir() for p in d.iterdir()
             if p.is_file() and p.suffix.lower() in VIDEO_TYPES]
    return sorted(found, key=lambda p: p.name.lower())


def find_video(recorded: str, run_dir: Path) -> str:
    """The video an analysis was made from. If it has been moved, look for the same file name in the usual places."""
    if Path(recorded).exists():
        return str(recorded)
    name = Path(recorded.replace("\\", "/")).name
    for d in (Path(run_dir), VIDEO_DIR, ROOT):
        if (d / name).exists():
            return str(d / name)
    return str(recorded)


def new_run_dir(runs: Path, video_name: str, start_s: float = 0.0, end_s: float | None = None) -> Path:
    """A fresh folder for a new analysis. Never reuses one: old names, ball results and pitch calibration would
    otherwise be applied to a different match."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video_name).stem).strip("_") or "match"
    if start_s or end_s is not None:
        base += f"_{int(start_s // 60)}to{'end' if end_s is None else int(-(-end_s // 60))}min"
    d, n = Path(runs) / base, 2
    while d.exists():
        d, n = Path(runs) / f"{base}_{n}", n + 1
    d.mkdir(parents=True)
    return d
