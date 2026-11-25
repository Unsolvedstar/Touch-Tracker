"""Drives the photo-grid naming in the app:  python make_demo.py && python test_grid.py"""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

f = Path("runs/_demo/names.json")
f.unlink(missing_ok=True)
at = AppTest.from_file("app.py", default_timeout=180)
at.session_state["run"] = "_demo"
at.run()
at.multiselect[0].set_value(list(at.multiselect[0].options)).run()
assert not at.exception, [e.value for e in at.exception]


def boxes():
    return [c for c in at.checkbox if c.key and c.key.startswith("pick_")]


def act(button, name=None):
    boxes()[0].check()
    if name:
        [t for t in at.text_input if t.label == "New player name"][0].set_value(name)
    [b for b in at.button if b.label == button][0].click().run()
    at.run()  # AppTest keeps a stale element right after st.rerun(); one more run clears it
    assert not at.exception, [e.value for e in at.exception]


n0 = len(boxes())
act("Name the ticked photos", "Sipho")
assert "Sipho" in json.loads(f.read_text(encoding="utf8")).values()
assert len(boxes()) == n0 - 1, (n0, len(boxes()))
act("Not our team")
assert "__not_ours__" in json.loads(f.read_text(encoding="utf8")).values()
assert len(boxes()) == n0 - 2, (n0, len(boxes()))
# second photo for the same player, picked from the dropdown of existing names
boxes()[0].check()
[s for s in at.selectbox if s.label.startswith("…or")][0].set_value("Sipho")
[b for b in at.button if b.label == "Name the ticked photos"][0].click().run()
at.run()
assert not at.exception, [e.value for e in at.exception]
assert len(boxes()) == n0 - 3, (n0, len(boxes()))
assert not at.error, [e.value for e in at.error]
print("OK", n0, "photos ->", len(boxes()), "unnamed;", f.read_text(encoding="utf8"))
f.unlink()
