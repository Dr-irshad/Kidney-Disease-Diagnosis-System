import os
import shutil
import numpy as np
from datetime import datetime
#from ruamel.yaml import YAML
from ultralytics import YOLO
from ultralytics import settings

if __name__ == '__main__':

    model = YOLO("yolov8n-seg.pt")  # load a pretrained YOLOv8n detection model

    config_dir = r"C:\Users\htluk\Documents\data\processed\yolov8\data_config.yaml"

    model.train(data=config_dir,
                epochs=50,
                batch=4,
                workers=2,
                seed=0,
                project=r".\model\yolov8",
                exist_ok=True,
                half=False, amp=False,
                imgsz=1080)  # train the model


