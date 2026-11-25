import re
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=180)
at.session_state["run"] = "_demo"
at.run()
assert not at.exception, at.exception
# find the cluster whose swatch is red
sw = [m.value for m in at.markdown if "background:rgb(" in m.value]
red = [i for i, v in enumerate(sw) if re.search(r"rgb\((1[5-9]\d|2\d\d),", v)]
print("swatches:", [re.search(r"rgb\([^)]*\)", v).group(0) for v in sw], "-> red idx", red)
at.multiselect[0].set_value([at.multiselect[0].options[i] for i in red]).run()
assert not at.exception, [e.value for e in at.exception]
print("caption:", [c.value for c in at.caption if "raw tracks" in c.value])
print("metrics:", [(m.label, m.value) for m in at.metric])
for d in at.dataframe[:3]:
    print(d.value.head(8).to_string()); print()
print("errors:", [e.value for e in at.error], "| warnings:", [w.value for w in at.warning])

# ---- pitch calibration + movement, then video render
use = [t for t in at.toggle if t.label == "Use this calibration"][0]
use.set_value(True).run()
assert not at.exception, [e.value for e in at.exception]
print("pitch charts:", len(at.get("plotly_chart")))
mv = [d.value for d in at.dataframe if "Distance (m)" in d.value.columns]
print(mv[0].head(3).to_string() if mv else "no movement table")
btn = [b for b in at.button if b.label == "Render annotated clip"][0]
btn.click().run()
assert not at.exception, [e.value for e in at.exception]
from pathlib import Path
p = Path("runs/_demo/annotated.mp4"); print("annotated.mp4 bytes:", p.stat().st_size)
