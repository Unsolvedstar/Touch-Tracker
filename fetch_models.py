"""Download everything the app would otherwise fetch on first use, so it works offline afterwards."""
from ultralytics import YOLO

import easyocr

YOLO("yolov8s.pt")  # default player detector (saved next to the app)
easyocr.Reader(["en"], gpu=False, verbose=False)  # shirt numbers
print("Models ready.")
