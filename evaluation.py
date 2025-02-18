import cv2
import os
from ultralytics import YOLO


model_path = r"/home/by4code/Documents/best.pt"

model = YOLO(model_path)

temp_img_path = r"/home/by4code/Documents/Test_images"
result = model(temp_img_path,
               conf=0.50,
               iou=0.5,
               agnostic_nms=True,
               save=True, save_txt=True, save_conf=True,
               imgsz=1920, line_width=1)

defect_count = 0
for counting in result:
    if counting.masks is not None:
        defect_count = defect_count+1
print(defect_count/len(result))
