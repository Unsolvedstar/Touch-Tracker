"""Pick a local video, analyse a few seconds, open the result:  python test_upload_flow.py  (takes a few minutes)"""
import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

import library

assert library.local_videos(), "put a video in videos/ first"
before = {p.name for p in Path("runs").iterdir()}
at = AppTest.from_file("app.py", default_timeout=900)
at.run()
assert not at.exception, [e.value for e in at.exception]
[n for n in at.number_input if n.label == "From (minute)"][0].set_value(2.5)
[n for n in at.number_input if n.label.startswith("How many minutes")][0].set_value(0.5)
at.run()
[b for b in at.button if b.label == "Analyse video"][0].click().run()
new = {p.name for p in Path("runs").iterdir()} - before
try:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(new) == 1, new
    assert any("Loaded" in s.value for s in at.success), [s.value for s in at.success]
    at.multiselect[0].set_value(list(at.multiselect[0].options)[:1]).run()
    assert not at.exception, [e.value for e in at.exception]
    print("OK", new, [s.value for s in at.success][0])
finally:
    for n in new:
        shutil.rmtree(Path("runs") / n, ignore_errors=True)
