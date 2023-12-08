from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from PIL import Image
import glob



model = YOLO('yolov8m.pt')
model.train(data='/workspace/SAFETY_DATA/data.yaml', epochs=2, patience=5, batch=8, imgsz=640)

# 훈련된 모델로 예측 수행
path_best_weights = "/workspace/runs/detect/train4/weights/best.pt"
model = YOLO(path_best_weights)
results = model.predict(source='/workspace/SAFETY_DATA/test/images',  # 데이터 디렉토리 변경
                        save=True)

# 예측된 이미지 시각화

os.makedirs('/workspace/working/result/',exist_ok=True)
predictions = glob.glob(os.path.join('/workspace/working/result/', 'runs/detect/predict', '*'))
rows, columns = 4, 4
total_images = rows * columns
num_images = min(total_images, len(predictions))
random_indices = np.random.choice(len(predictions), num_images, replace=False)

fig, axes = plt.subplots(rows, columns, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    if i < num_images:
        idx = random_indices[i]
        image_path = predictions[idx]
        image = Image.open(image_path)

        # 이미지를 플로팅
        ax.imshow(image)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()