import torch
from ultralytics import YOLO

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
model = YOLO("yolo11x.pt")
model.train(data="datasets/ncert/data.yaml", epochs=50, imgsz=640, device=device)
