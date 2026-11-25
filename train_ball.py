"""Train the small-ball detector.   python train_ball.py DATASET_YAML [--epochs 15] [--base yolov8n.pt]

Works on the synthetic set from make_ball_dataset.py or on a real labelled set (e.g. a Roboflow export, one class
'ball'); with real data, pass it instead of / merged with the synthetic one. On a GPU (Colab) use --device 0 and
more epochs. The best weights are copied to models/ball.pt, which ball.py picks up automatically."""
import argparse, shutil
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--base", default="yolov8n.pt")
ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr0", type=float, default=0.01); ap.add_argument("--device", default="cpu"); ap.add_argument("--name", default="ball")
a = ap.parse_args()
model = YOLO(a.base)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=0, cache="ram",
            project=str(Path(__file__).parent / "runs_train"), name=a.name, exist_ok=True,
            scale=0.3, mosaic=0.3, degrees=0, flipud=0.0, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.35,
            lr0=a.lr0, single_cls=True, patience=8, plots=False, verbose=True)
best = Path(model.trainer.save_dir) / "weights" / "best.pt"
Path(__file__).parent.joinpath("models").mkdir(exist_ok=True)
shutil.copy(best, Path(__file__).parent / "models" / "ball.pt")
print("saved models/ball.pt")
