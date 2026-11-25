# ⚽ Touch Tracker

Free football stats from phone video, for small teams: touches, passes, time on the ball, heatmaps and movement.
It runs on your own computer. Your video is not uploaded anywhere.

## Install (once)

Get the app: on this page click **Code → Download ZIP**, then unzip it.

1. Install [Python 3.10 or newer](https://www.python.org/downloads/) and tick **Add python.exe to PATH**.
2. Double-click **setup.bat**. It downloads several hundred MB, so use Wi-Fi if you can.

After that the app works without internet.

## Use

1. Copy your match video into the **videos** folder.
2. Double-click **run.bat**. The app opens in your browser. Keep the black window open.
3. **Analyse a video.** Start with 2 minutes. On a laptop without a graphics card, allow roughly 5–8 minutes of
   processing for each minute of football, and keep it plugged in.
4. **Pick your team's kit colour.**
5. **Name your players:** tick every photo of one player and give them a name. Mark opponents and referees as
   *Not our team*.
6. **Results:** summary, passes, pitch maps, an annotated clip, and CSV downloads.
7. Optional, on *Pitch & movement*: mark four corners of a known rectangle to get distances in metres.

## How far to trust the numbers

These are **estimates** to help spot patterns, not an official record.

* The ball is tiny in phone video. On our test match it was found in roughly 30–60% of frames, and some of those
  detections were boots or spare balls. Touches and passes are therefore missed or credited to the wrong player.
* Players leave the picture when the camera pans, and come back as "new" people until you name them.
* Shirt numbers are only readable when a player is close and facing away.
* Distances depend on how carefully the pitch was marked, and get worse the further the camera swings from that view.
* The ball model was trained on one amateur match, so it may do worse with a different ball, pitch or light.
* Accuracy has not been measured against a hand-counted match. Check a few events yourself before relying on it.

## Filming for good results

Zoom in on the play, film from as high as you can near halfway, use 1080p landscape, pan smoothly, and keep spare
balls away from the touchline. The app has the full checklist.

## Players' privacy (POPIA)

Match video and player names are personal information.

* Get players' agreement before filming and naming them; for under-18s, a parent or guardian's.
* Videos, photos and names stay in the **videos** and **runs** folders on your computer. Delete a folder to delete the data.
* Don't share clips or player stats without permission.

## Problems

| What you see | What to do |
|---|---|
| "Python was not found" | Reinstall Python with *Add python.exe to PATH* ticked. |
| Ball found in very few frames | Film closer, or try a larger detector in the sidebar (slower). |
| Far more than 11 players | Name them in step 2: tracks with one name are counted as one player. |
| "The video can't be found" | Copy it into **videos** with the same file name. |

## For developers

`python run_tests.py` (add `--video` to analyse a real clip). `train_ball.py` retrains the ball detector.

## Licence

[AGPL-3.0](LICENSE). The app uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed the
same way: if you distribute or host a changed version, publish its source too, or get a commercial Ultralytics licence.
