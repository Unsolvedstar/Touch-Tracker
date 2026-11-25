import re, sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=600)
at.session_state["run"] = (sys.argv[2] if len(sys.argv) > 2 else "phone_match")
at.run()
assert not at.exception, [e.value for e in at.exception]
sw = [re.search(r"rgb\([^)]*\)", m.value).group(0) + " | " + c.value for m, c in zip([m for m in at.markdown if "background:rgb(" in m.value], [c for c in at.caption if c.value.startswith("Cluster")])]
print("clusters:", *sw, sep="\n  ")
print("info:", [i.value for i in at.info][:3]); print("warnings:", [w.value for w in at.warning])

home = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ms = at.multiselect[0]
ms.set_value([ms.options[home]]).run()
assert not at.exception, [e.value for e in at.exception]
print("caption:", [c.value[:90] for c in at.caption if "raw tracks" in c.value])
print("metrics:", [(m.label, m.value) for m in at.metric])
for d in at.dataframe[:3]:
    print(d.value.head(12).to_string()); print()
print("warnings:", [w.value for w in at.warning], "| errors:", [e.value for e in at.error])
