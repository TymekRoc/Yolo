import os
import torch
from pathlib import Path

# Ustawienia
IMG_SIZE = 640  # rozmiar obrazu wejściowego
BATCH_SIZE = 16  # rozmiar batcha
EPOCHS = 50  # liczba epok treningowych
MODEL = 'yolov5s.pt'  # pretrenowany model YOLOv5 (możesz użyć np. yolov5s, yolov5m, yolov5l, yolov5x)
DATA_CONFIG = 'drone.yaml'  # plik YAML opisujący dane

# Sprawdzenie dostępności GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on {device}')

# Ścieżka do skryptu treningowego
script_path = Path('train.py')

# Komenda treningowa
os.system(f'python3 {script_path} --img {IMG_SIZE} --batch {BATCH_SIZE} --epochs {EPOCHS} '
          f'--data {DATA_CONFIG} --weights {MODEL} --device {device}')
