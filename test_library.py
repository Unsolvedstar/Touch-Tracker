import tempfile
from pathlib import Path

import library

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    library.ROOT, library.VIDEO_DIR = root, root / "videos"
    library.VIDEO_DIR.mkdir()
    (library.VIDEO_DIR / "Match 1.MP4").write_bytes(b"x")
    (root / "notes.txt").write_bytes(b"x")
    assert [p.name for p in library.local_videos()] == ["Match 1.MP4"]

    # a moved video is found again by name; a truly missing one is returned unchanged
    run = root / "runs" / "a"
    assert library.find_video("C:\\old\\place\\Match 1.MP4", run) == str(library.VIDEO_DIR / "Match 1.MP4")
    assert library.find_video("C:/old/gone.mp4", run) == "C:/old/gone.mp4"

    # a second analysis of the same file never reuses the first folder
    a = library.new_run_dir(root / "runs", "Match 1.MP4")
    b = library.new_run_dir(root / "runs", "Match 1.MP4")
    assert (a.name, b.name) == ("Match_1", "Match_1_2")
    assert library.new_run_dir(root / "runs", "Match 1.MP4", 120, 270).name == "Match_1_2to5min"
print("OK")
