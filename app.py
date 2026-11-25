import base64
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ball import CUSTOM_WEIGHTS, run_ball_pass
from detect import load_run, probe, run_detection
from library import VIDEO_DIR, local_videos, new_run_dir
from passes import detect_passes, pass_matrix, pass_summary
from pitch import (PRESETS, heatmap_figure, homography, movement_stats, network_figure, player_positions,
                   to_pitch, touches_figure)
from possession import OTHER, PossessionParams, compute_possession, filter_ball_candidates
from render import render_annotated
from teams import cluster_tracks
from guide import FILMING_TIPS
from identity import NOT_OURS, assign, load_names, merge_keys, on_screen_twice, root_names, save_names
from tracks import stitch_tracks

st.set_page_config(page_title="Touch Tracker", page_icon="⚽", layout="wide")
RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


@st.cache_data(show_spinner=False)
def video_frame(path: str, idx: int = 0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return img[:, :, ::-1].copy() if ok else None


@st.cache_data(show_spinner=False)
def crop_data_uri(path: str) -> str | None:
    """Small inline thumbnail so each track can be recognised in the table."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (64, int(64 * img.shape[0] / img.shape[1])))
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def root_crop(run_dir: Path, member_ids) -> str | None:
    for m in member_ids:
        p = run_dir / "crops" / f"{m}_0.jpg"
        if p.exists():
            return crop_data_uri(str(p))
    return None


st.title("⚽ Touch Tracker")
st.caption("Free football stats from phone video: touches, passes, time on the ball and movement for your players.")

# ───────────────────────── video ─────────────────────────
with st.sidebar:
    st.header("Processing")
    auto = st.checkbox("Auto frame sampling (recommended)", value=True,
                       help="Analyses about 8 frames per second whatever the phone's frame rate (30 or 60 fps).")
    stride = None if auto else st.select_slider("Analyse every Nth frame", [1, 2, 3, 4, 6, 8], value=3,
                                                 help="Higher = faster on CPU, but short touches can be missed.")
    model_name = st.selectbox("Detector", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"], index=1,
                              help="yolov8s is a good balance on a laptop. Bigger is slower; smaller misses more players.")
    imgsz = st.select_slider("Detection resolution", [640, 960, 1280, 1600], value=1280)
    deep_ball = st.checkbox("Deep ball search (recommended)", value=True,
                            help="Searches around the players at full resolution for the small ball and follows it through "
                                 "time. About 1 extra second per analysed frame on CPU.")

with st.expander("🎥 How to film for useful stats (read before match day)"):
    st.markdown(FILMING_TIPS)
st.caption("🔒 Your video is processed on this computer and is not uploaded anywhere. Only film and name players who have "
           "agreed to it (for under-18s, a parent or guardian), and delete footage you no longer need.")


def analyse(source: Path | None, upload, start_s: float, end_s: float | None):
    """Run detection (and the ball search) into a fresh folder. On failure the half-made folder is removed."""
    video_name = upload.name if upload else source.name
    d = new_run_dir(RUNS, video_name, start_s, end_s)
    try:
        if upload:
            source = d / Path(upload.name).name
            upload.seek(0)
            with open(source, "wb") as out:
                shutil.copyfileobj(upload, out)
        bar = st.progress(0.0, text="Step 1 of 2 · finding and tracking players… (leave this tab open)")
        run_detection(str(source.resolve()), d, stride, model_name, imgsz, progress=lambda f: bar.progress(f),
                      start_s=start_s, end_s=end_s)
        bar.empty()
        if pd.read_csv(d / "players.csv").empty:
            raise RuntimeError("No players were found in that part of the video.")
        if deep_ball:
            bar = st.progress(0.0, text="Step 2 of 2 · searching for the ball…")
            run_ball_pass(d, progress=lambda f: bar.progress(f))
            bar.empty()
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        st.error(f"The analysis didn't finish: {e}")
        st.stop()
    st.session_state["run"] = d.name


existing = sorted(p.name for p in RUNS.iterdir() if (p / "meta.json").exists() and not p.name.startswith("_"))
tab_new, tab_old = st.tabs(["Analyse a video", "Open earlier analysis"])
with tab_new:
    local = local_videos()
    how = st.radio("Where is the video?", ["On this computer (videos folder)", "Upload it"], horizontal=True,
                   index=0 if local else 1)
    source, up = None, None
    if how.startswith("On"):
        if local:
            source = st.selectbox("Video", local, format_func=lambda p: f"{p.name}  ({p.stat().st_size / 1e6:,.0f} MB)")
        else:
            st.info(f"Copy your match video into this folder, then reload the page:  {VIDEO_DIR}")
    else:
        up = st.file_uploader("Match video", type=["mp4", "mov", "avi", "mkv"],
                              help="For big files, copying into the videos folder is faster and saves disk space.")
    c1, c2 = st.columns(2)
    from_min = c1.number_input("From (minute)", 0.0, 200.0, 0.0, 0.5)
    mins = c2.number_input("How many minutes to analyse (0 = to the end)", 0.0, 200.0, 2.0, 0.5)
    length = mins or None
    if source is not None:
        info = probe(str(source))
        total = info["frames"] / info["fps"] / 60 if info["fps"] > 0 else 0
        length = max(min(mins or total, total - from_min), 0)
        st.caption(f"This video is {total:.1f} minutes long.")
    if length:
        st.caption(f"⏱ Expect roughly {length * 5:.0f}–{length * 8:.0f} minutes of processing on a laptop without a "
                   "graphics card. Start with 2 minutes to check the results are useful. Keep the laptop plugged in.")
    if (source is not None or up is not None) and st.button("Analyse video", type="primary"):
        if source is not None and length == 0:
            st.error("'From' is past the end of the video.")
        else:
            analyse(source, up, from_min * 60, (from_min + mins) * 60 if mins else None)
with tab_old:
    if existing:
        pick = st.selectbox("Analysis", existing)
        if st.button("Open"):
            st.session_state["run"] = pick
    else:
        st.info("Nothing analysed yet.")

if "run" not in st.session_state:
    st.stop()
run_name = st.session_state["run"]
run_dir = RUNS / run_name
players, ball, meta, cam = load_run(run_dir)
dt = meta["dt"]
if not Path(meta["video"]).exists():
    st.error(f"The video for this analysis can't be found: {meta['video']}. Copy it into {VIDEO_DIR} "
             "(same file name) and reload.")
    st.stop()
if not (run_dir / "ball_candidates.csv").exists():
    if st.button("🔎 Run deep ball search on this analysis (slower, finds small balls)"):
        bar = st.progress(0.0, text="Searching for the ball…")
        run_ball_pass(run_dir, progress=lambda f: bar.progress(f))
        bar.empty()
        st.rerun()
else:
    st.caption("Ball search: " + ("custom small-ball model" if CUSTOM_WEIGHTS.exists() else "general detector (no custom ball model installed)")
               + ". Re-run it after installing a new ball model by deleting ball_candidates.csv in the run folder.")
ball, ignored = filter_ball_candidates(ball, players)  # spare balls on the touchline, etc.
st.success(f"Loaded **{run_name}**: {players['track_id'].nunique()} player tracks, "
           f"ball seen in {ball['frame'].nunique()} of {players['frame'].nunique()} analysed frames.")
moved, bad = cam.pan_degrees(), 1 - cam.ok_fraction
if moved > 3:
    st.info(f"Moving camera detected (it swung about {moved:.0f}° from where it started). Camera-motion compensation is on, "
            "so touches, heatmaps and distances are measured against the pitch, not the screen."
            + (f" Camera tracking lost lock on {bad:.0%} of frames; those moments are treated as steady." if bad > 0.05 else "")
            + ("" if cam.f_estimated else " (Little camera movement to learn the lens from, so a typical phone lens was assumed.)"))
if ignored["oversized"] or ignored["static"]:
    st.caption(f"Ignored {ignored['oversized']} oversized and {ignored['static']} never-moving ball detections "
               "(e.g. spare balls lying beside the pitch).")
seen = ball["frame"].nunique() / max(players["frame"].nunique(), 1)
if seen < 0.2:
    st.warning(f"The match ball was found in only {seen:.0%} of frames, so touches, passes and time on the ball will be "
               "unreliable or missing. The ball is tiny in wide shots: try a bigger detector or higher detection resolution, "
               "or film closer / zoom in on the play.")

# ───────────────────────── 1. home team ─────────────────────────
st.header("1 · Which kit is your team?")
track_cluster, ctable = cluster_tracks(players)
cols = st.columns(len(ctable) or 1)
for c, (_, r) in zip(cols, ctable.iterrows()):
    c.markdown(f"<div style='height:42px;border-radius:8px;background:rgb({r.r},{r.g},{r.b})'></div>",
               unsafe_allow_html=True)
    c.caption(f"Cluster {r.cluster} · {r.tracks} tracks")
home_clusters = st.multiselect("Select your team's cluster(s) (add the goalkeeper's if separate)",
                               list(ctable["cluster"]))
home_orig = sorted(t for t, c in track_cluster.items() if c in home_clusters)
if not home_orig:
    st.stop()

# ───────────────────────── 2. identities ─────────────────────────
st.header("2 · Who is who?")
stitch = st.toggle("Auto-merge fragmented tracks", value=True,
                   help="Links a track that ends to one that starts nearby a moment later (same player, new tracker ID).")
reach = st.slider("Merge reach (seconds a player can be out of view and still be re-linked)", 1.0, 20.0, 6.0, 0.5,
                  disabled=not stitch,
                  help="Handheld footage pans away from players, so they vanish and come back under a new ID. "
                       "Higher = fewer duplicate players but more risk of joining two different people.")
ocr_file = run_dir / "numbers.csv"
trusted: dict[int, int] = {}  # shirt numbers read from >= 3 agreeing crops: safe enough to merge on
if ocr_file.exists():
    _o = pd.read_csv(ocr_file)
    if "votes" in _o:
        _o = _o[_o["number"].notna() & (_o["votes"] >= 3) & (_o["ocr_conf"] >= 0.5)]
        trusted = {int(t): int(n) for t, n in zip(_o["track_id"], _o["number"])}
use_numbers = st.toggle("Use shirt numbers to merge tracks", value=bool(trusted), disabled=not trusted, help=(
    "Joins tracks of the same number across any gap (if never on screen together) and never joins different numbers. "
    "Needs shirt numbers to have been read (OCR) from at least 3 agreeing clear photos of a track." if trusted else
    "Read the shirt numbers first (button below, then reload)."))
saved_names = load_names(run_dir)
home_set = set(home_orig)
named = {t: n for t, n in saved_names.items() if t in home_set}
roots = (stitch_tracks(players, home_orig, dt, max_gap_s=reach,
                       numbers=merge_keys(named, trusted if use_numbers else {}))
         if stitch else {t: t for t in home_orig})
members: dict[int, list[int]] = {}
for t, r in roots.items():
    members.setdefault(r, []).append(t)
home_roots = sorted(members)
st.caption(f"{len(home_orig)} raw tracks → {len(home_roots)} players after merging.")

P = players.copy()  # same data, home tracks re-keyed to their merged root id
P["track_id"] = P["track_id"].map(roots).fillna(P["track_id"]).astype(int)

if st.button("Read shirt numbers (OCR)"):
    from jersey import read_numbers

    bar = st.progress(0.0, text="Reading numbers…")
    res = read_numbers(run_dir / "crops", home_orig, progress=lambda f: bar.progress(f))
    bar.empty()
    pd.DataFrame([(t, *v) for t, v in res.items()],
                 columns=["track_id", "number", "ocr_conf", "votes", "crops_read"]).to_csv(ocr_file, index=False)
    st.rerun()

ocr = pd.read_csv(ocr_file).set_index("track_id") if ocr_file.exists() else pd.DataFrame(columns=["number", "ocr_conf"])
frames_seen = P.groupby("track_id")["frame"].nunique()
name_of_root = root_names(members, named)
by_seen = sorted(home_roots, key=lambda r: -frames_seen.get(r, 0))

# ── tap-the-photos naming ──
st.subheader("Tick every photo of one player, then name them")
st.caption("Longest-seen players come first, so naming the top of the list covers the most. Names are saved with "
           "this analysis and tracks given the same name are merged.")
chosen_names = sorted({n for n in saved_names.values() if n != NOT_OURS})
n_unnamed = sum(r not in name_of_root for r in home_roots)
secs_total = max(float(sum(frames_seen.get(r, 0) for r in home_roots if name_of_root.get(r) != NOT_OURS)), 1.0)
secs_named = sum(frames_seen.get(r, 0) for r in home_roots if name_of_root.get(r, NOT_OURS) != NOT_OURS)
st.progress(min(secs_named / secs_total, 1.0),
            text=f"{secs_named / secs_total:.0%} of your team's on-screen time is named · {n_unnamed} photos still unnamed")
g1, g2 = st.columns([3, 1])
show = g1.selectbox("Show", ["Unnamed", "Everyone", "Not our team"] + chosen_names, key=f"show_{run_name}")
pool = [r for r in by_seen if
        (show == "Everyone") or (show == "Unnamed" and r not in name_of_root)
        or (show == "Not our team" and name_of_root.get(r) == NOT_OURS) or name_of_root.get(r) == show]
PER_PAGE, PER_ROW = 24, 6
pages = max((len(pool) + PER_PAGE - 1) // PER_PAGE, 1)
page = int(g2.number_input(f"Page (of {pages})", 1, pages, 1, key=f"page_{run_name}_{show}_{pages}"))
shown = pool[(page - 1) * PER_PAGE: page * PER_PAGE]
if not shown:
    st.info("No photos in this view.")
else:
    with st.form(f"grid_{run_name}", clear_on_submit=True):
        picked = []
        for start in range(0, len(shown), PER_ROW):
            for col, r in zip(st.columns(PER_ROW), shown[start:start + PER_ROW]):
                photo = next((p for m in sorted(members[r]) for p in [run_dir / "crops" / f"{m}_0.jpg"]
                              if p.exists()), None)
                if photo:
                    col.image(str(photo), width="stretch")
                else:
                    col.caption("(no photo)")
                who = name_of_root.get(r)
                label = "not ours" if who == NOT_OURS else who or f"Track {r}"
                if col.checkbox(f"{label} · {frames_seen.get(r, 0) * dt:.0f}s", key=f"pick_{run_name}_{r}"):
                    picked.append(r)
        n1, n2 = st.columns(2)
        typed = n1.text_input("New player name", placeholder="e.g. Sipho or #10")
        existing_name = n2.selectbox("…or a player you've already named", chosen_names, index=None)
        new_name = typed.strip() or existing_name  # a typed name wins
        b1, b2, b3 = st.columns(3)
        do_name = b1.form_submit_button("Name the ticked photos", type="primary")
        do_out = b2.form_submit_button("Not our team")
        do_clear = b3.form_submit_button("Remove name")
    if do_name or do_out or do_clear:
        if not picked:
            st.warning("Tick at least one photo first.")
        elif do_name and not (new_name or "").strip():
            st.warning("Choose or type a name first, then tick the photos again.")
        else:
            ids = [t for r in picked for t in members[r]]
            save_names(run_dir, assign(saved_names, ids, NOT_OURS if do_out else new_name if do_name else None))
            st.rerun()
for who, n in on_screen_twice(players, named).items():
    st.warning(f"**{who}** appears twice at the same moment ({n * dt:.1f}s in total), so at least one of those photos is "
               f"someone else. Choose '{who}' under Show and remove the wrong one.")

rows = []
for r in home_roots:
    if name_of_root.get(r) == NOT_OURS:
        continue
    cand = ocr.reindex(members[r]).dropna(subset=["number"])
    best = cand.sort_values("ocr_conf", ascending=False).iloc[0] if len(cand) else None
    rows.append(dict(photo=root_crop(run_dir, sorted(members[r])), player_id=r,
                     merged_tracks=", ".join(map(str, sorted(members[r]))),
                     seconds_seen=round(frames_seen.get(r, 0) * dt, 1),
                     ocr_conf=None if best is None else float(best["ocr_conf"]),
                     number=pd.NA if best is None else int(best["number"]), player=name_of_root.get(r, "")))
tbl = pd.DataFrame(rows, columns=["photo", "player_id", "merged_tracks", "seconds_seen", "ocr_conf", "number",
                                  "player"]).astype({"number": "Int64"})
tbl = tbl.sort_values("seconds_seen", ascending=False).reset_index(drop=True)
left, right = st.columns([3, 2])
with left:
    st.markdown("**Fine-tune (optional).** Fix a shirt number or name here. Rows with the same name or number are "
                "counted as one player. Edits in this table last for this session; the photo grid above is what gets saved.")
    tbl = st.data_editor(tbl, hide_index=True, row_height=64,
                         disabled=["photo", "player_id", "merged_tracks", "seconds_seen", "ocr_conf"],
                         column_config={"photo": st.column_config.ImageColumn("Photo", width="small"),
                                        "player": st.column_config.TextColumn("Player name"),
                                        "number": st.column_config.NumberColumn("Shirt #", min_value=0, max_value=99, step=1)},
                         width="stretch", key=f"tracks_{run_name}_{stitch}_{len(home_roots)}_{hash(tuple(sorted(named.items())))}")
with right:
    st.markdown("**Roster** (number → name)")
    roster = st.data_editor(pd.DataFrame({"number": pd.Series(dtype="Int64"), "name": pd.Series(dtype="str")}),
                            num_rows="dynamic", hide_index=True, width="stretch", key="roster")
names = {int(r.number): r.name for r in roster.dropna().itertuples() if str(r.name).strip()}

label_of: dict[int, str] = {}
for r in tbl.itertuples():
    if str(r.player).strip():                       # a name typed by the coach wins
        label_of[r.player_id] = str(r.player).strip()
    elif pd.notna(r.number):
        n = int(r.number)
        label_of[r.player_id] = f"#{n} {names[n]}" if n in names else f"#{n}"
    else:
        label_of[r.player_id] = f"Track {r.player_id}"

# ───────────────────────── 3. analysis ─────────────────────────
st.header("3 · Results")
st.info(f"**Treat these as estimates, not an official record.** The ball was found in {seen:.0%} of frames and some of "
        "those detections are wrong, so touches and passes are often missed or credited to the wrong player. Use them "
        "to spot patterns, and check anything that matters against the video ('Check the work' tab).")
with st.expander("Tuning (defaults are a sensible start)"):
    c1, c2, c3, c4 = st.columns(4)
    radius = c1.slider("Control radius (× player height)", 0.2, 1.5, 0.6, 0.05,
                       help="How close the ball must be to a player's feet to count as theirs. Wider suits small, distant, jittery ball detections.")
    dv = c2.slider("Touch sensitivity: ball velocity change", 0.5, 10.0, 2.0, 0.5,
                   help="Lower = more touches counted inside a possession.")
    min_spell = c3.slider("Minimum possession (s)", 0.1, 1.0, 0.3, 0.1)
    pass_gap = c4.slider("Max ball-travel time for a pass (s)", 1.0, 6.0, 3.0, 0.5)
params = PossessionParams(radius_factor=radius, touch_dv_h_s=dv, min_spell_s=min_spell)

summary, events, touches = compute_possession(P, ball, label_of, dt, params)
if summary.empty:
    st.info("No possession spells found. Try a larger control radius or check ball detection.")
    st.stop()
home_players = sorted(set(label_of.values()))
if len(home_players) > 14:
    st.warning(f"{len(home_players)} different home players are being tracked, but a team has 11 on the pitch. The same player "
               "appearing under several tracks splits their touches and creates false passes (them 'passing' to themselves). "
               "Merge duplicates in step 2: tick all their photos and give them one name.")
passes = detect_passes(events, pass_gap)
psum = pass_summary(passes, list(summary["player"]))
pmat = pass_matrix(passes, list(summary["player"]))

t_sum, t_pass, t_pitch, t_video, t_check = st.tabs(
    ["📊 Summary", "🔁 Passes", "🗺️ Pitch & movement", "🎬 Annotated video", "🔍 Check the work"])
# Tabs are filled out of display order below: the pitch calibration feeds the pass network and the summary report.

H = None
movement = None
Lm = Wm = None
full = False
with t_pitch:
    st.caption("Pick a moment where the pitch markings are clearly visible, then enter the pixel position of four corners of "
               "a known rectangle on the pitch. The camera's movement is tracked from that moment, so a panning phone "
               "camera is fine. Positions get less accurate the further the camera has swung from the calibrated view.")
    calib_f = run_dir / "calib.json"
    saved = json.loads(calib_f.read_text()) if calib_f.exists() else {}
    preset = st.selectbox("Reference rectangle", list(PRESETS), index=list(PRESETS).index(saved["preset"])
                          if saved.get("preset") in PRESETS else 0)
    Lm, Wm, full = PRESETS[preset]
    if preset.startswith("Custom"):
        cc1, cc2 = st.columns(2)
        Lm = cc1.number_input("Length (m, left→right)", 1.0, 120.0, float(saved.get("L", Lm)))
        Wm = cc2.number_input("Width (m, top→bottom)", 1.0, 120.0, float(saved.get("W", Wm)))
    duration = float(meta["frames"] / meta["fps"])
    calib_t = st.slider("Calibration moment (s)", 0.0, max(duration - 0.1, 0.1), float(saved.get("t", 0.0)), 0.5)
    c_frame = (int(calib_t * meta["fps"]) // meta["stride"]) * meta["stride"]  # nearest analysed frame
    frame0 = video_frame(meta["video"], c_frame)
    if frame0 is None:
        st.error("Couldn't read the video file to show a calibration frame.")
    else:
        fh, fw = frame0.shape[:2]
        dflt = saved.get("pts") or [[0.15 * fw, 0.25 * fh], [0.85 * fw, 0.25 * fh], [0.95 * fw, 0.9 * fh],
                                    [0.05 * fw, 0.9 * fh]]
        pts = []
        pcols = st.columns(4)
        for i, nm in enumerate(["Top-left", "Top-right", "Bottom-right", "Bottom-left"]):
            with pcols[i]:
                x = st.number_input(f"{nm} x", -fw, 2 * fw, int(dflt[i][0]), key=f"cx{i}_{run_name}_{c_frame}")
                y = st.number_input(f"{nm} y", -fh, 2 * fh, int(dflt[i][1]), key=f"cy{i}_{run_name}_{c_frame}")
                pts.append((x, y))
        prev = frame0.copy()
        cv2.polylines(prev, [np.array(pts, np.int32)], True, (255, 255, 0), max(fw // 400, 2))
        for i, (x, y) in enumerate(pts):
            cv2.circle(prev, (int(x), int(y)), max(fw // 120, 5), (255, 0, 0), -1)
            cv2.putText(prev, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, max(fw / 1000, 0.7),
                        (255, 255, 0), 2)
        st.image(prev, caption="Yellow outline = the reference rectangle as you've placed it (order: 1 top-left, "
                 "2 top-right, 3 bottom-right, 4 bottom-left)", width="stretch")
        b1, b2 = st.columns(2)
        use = b1.toggle("Use this calibration", value=bool(saved))
        if b2.button("Save calibration for this video"):
            calib_f.write_text(json.dumps(dict(preset=preset, L=Lm, W=Wm, t=calib_t, pts=[list(p) for p in pts])))
            st.toast("Saved")
        if use:
            H = cam.pitch_mapper(homography(pts, Lm, Wm), c_frame)  # stabilised coords -> pitch metres

    if H is not None:
        pos = player_positions(P, label_of, H)
        movement = movement_stats(pos)
        who = st.selectbox("Player", ["All home players"] + list(summary["player"]))
        sel = pos if who.startswith("All") else pos[pos["player"] == who]
        ts = touches if who.startswith("All") else touches[touches["player"] == who]
        ts = ts[ts["player"].isin(home_players)]
        tx, ty = H(ts["sx"], ts["sy"]) if len(ts) else (np.array([]), np.array([]))
        g1, g2 = st.columns(2)
        g1.plotly_chart(heatmap_figure(sel["X"], sel["Y"], Lm, Wm, full, f"Position heatmap · {who}"), width="stretch")
        g2.plotly_chart(touches_figure(tx, ty, Lm, Wm, full, f"Where touches happened · {who} ({len(ts)})"),
                        width="stretch")
        st.subheader("Movement")
        st.dataframe(movement.rename(columns={"player": "Player", "distance_m": "Distance (m)",
                                              "minutes_tracked": "Minutes tracked", "avg_speed_kmh": "Avg speed (km/h)",
                                              "top_speed_kmh": "Top speed (km/h)"}), hide_index=True, width="stretch")
        st.caption("Distance only counts time the player was tracked, so it under-reports if they were hidden or the "
                   "ID was lost. Steps faster than 11 m/s are discarded as tracking glitches.")
    else:
        st.info("Turn on “Use this calibration” once the yellow outline sits on the pitch markings.")

with t_pass:
    st.caption("A pass = the ball goes from one home player's possession to another's within the max travel time. "
               "Home → opposition is a lost ball; opposition → home is a ball won. Shots and clearances that leave play "
               "aren't counted.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Completed passes", int((passes["outcome"] == "completed").sum()))
    m2.metric("Balls lost", int((passes["outcome"] == "lost").sum()))
    m3.metric("Balls won", int((passes["outcome"] == "won").sum()))
    st.dataframe(psum.rename(columns={"player": "Player", "passes_completed": "Passes completed", "ball_lost": "Ball lost",
                                      "pass_completion_pct": "Completion %", "received": "Received", "ball_won": "Ball won"}),
                 hide_index=True, width="stretch")
    if pmat.values.sum():
        st.plotly_chart(px.imshow(pmat, text_auto=True, color_continuous_scale="Greens", aspect="auto",
                                  title="Completed passes (row = passer, column = receiver)"), width="stretch")
        if H is not None:
            avg = {p: (g["X"].mean(), g["Y"].mean()) for p, g in pos.groupby("player")}
            st.plotly_chart(network_figure(avg, pmat, Lm, Wm, full), width="stretch")
    pl = passes.assign(time=passes["t"].map(mmss))[["time", "from", "to", "outcome", "gap_s"]]
    with st.expander("Every pass / turnover"):
        st.dataframe(pl, hide_index=True, width="stretch")
    st.download_button("Download passes CSV", pl.to_csv(index=False), "passes.csv", "text/csv")

with t_sum:
    total_s = players["frame"].nunique() * dt
    m1, m2, m3 = st.columns(3)
    m1.metric("Total touches (home)", int(summary["touches"].sum()))
    m2.metric("Home time on ball", f"{summary['time_on_ball_s'].sum():.0f} s")
    m3.metric("Footage analysed", f"{total_s / 60:.1f} min")
    report = summary.merge(psum, on="player", how="left")
    if movement is not None:
        report = report.merge(movement, on="player", how="left")
    show = report.rename(columns={"player": "Player", "touches": "Touches", "time_on_ball_s": "Time on ball (s)",
                                  "spells": "Possessions", "longest_spell_s": "Longest (s)",
                                  "passes_completed": "Passes completed", "ball_lost": "Ball lost",
                                  "pass_completion_pct": "Pass %", "received": "Received", "ball_won": "Ball won",
                                  "distance_m": "Distance (m)", "minutes_tracked": "Minutes tracked",
                                  "avg_speed_kmh": "Avg km/h", "top_speed_kmh": "Top km/h"})
    st.dataframe(show, hide_index=True, width="stretch")
    st.download_button("Download full report CSV", show.to_csv(index=False), "player_report.csv", "text/csv")
    g1, g2 = st.columns(2)
    g1.plotly_chart(px.bar(summary, x="player", y="touches", title="Touches"), width="stretch")
    g2.plotly_chart(px.bar(summary, x="player", y="time_on_ball_s", title="Time on ball (s)"), width="stretch")

with t_video:
    st.caption("Renders the footage with names, who is on the ball (yellow), the ball (red) and a live touch counter. "
               "Rendering is CPU-bound, so do a short range first.")
    v1, v2 = st.columns(2)
    start_s = v1.number_input("Start (s)", 0.0, float(meta["frames"] / meta["fps"]), 0.0, 5.0)
    length_s = v2.number_input("Length (s)", 5.0, 300.0, 30.0, 5.0)
    if st.button("Render annotated clip"):
        bar = st.progress(0.0, text="Rendering…")
        out, playable = render_annotated(meta, P, ball, label_of, events, touches, run_dir / "annotated.mp4",
                                         start_s, start_s + length_s, progress=lambda f: bar.progress(min(f, 1.0)))
        bar.empty()
        st.session_state["rendered"] = (str(out), playable)
    if "rendered" in st.session_state and Path(st.session_state["rendered"][0]).exists():
        out, playable = st.session_state["rendered"]
        if playable:
            st.video(out)
        else:
            st.info("No ffmpeg available, so the clip can't play in the browser. Download it to watch.")
        st.download_button("Download clip", Path(out).read_bytes(), "annotated.mp4", "video/mp4")

with t_check:
    st.caption("Spot-check events against the video before trusting the totals.")
    ev = events[events["player"] != OTHER].reset_index(drop=True)
    ev["start"] = ev["start_t"].map(mmss)
    st.dataframe(ev[["player", "start", "duration_s", "touches"]], hide_index=True, width="stretch", height=240)
    if len(ev):
        i = st.number_input("Event row to preview", 0, len(ev) - 1, 0)
        row = ev.iloc[int(i)]
        img = video_frame(meta["video"], int(row.start_frame))
        if img is not None:
            img = img[:, :, ::-1].copy()
            for r in P[P["frame"] == row.start_frame].itertuples():
                lab = label_of.get(r.track_id)
                colr = (0, 200, 0) if lab == row.player else (150, 150, 150)
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colr, 2 if lab == row.player else 1)
                if lab == row.player:
                    cv2.putText(img, lab, (int(r.x1), int(r.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colr, 2)
            for r in ball[ball["frame"] == row.start_frame].itertuples():
                cv2.circle(img, (int((r.x1 + r.x2) / 2), int((r.y1 + r.y2) / 2)), 12, (0, 0, 255), 2)
            st.image(img[:, :, ::-1], caption=f"{row.player} at {row.start}", width="stretch")
