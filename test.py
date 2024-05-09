from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predtictor = DefaultPredictor(cfg)


image_path = "test\Image_70.jpg"
#videoPath = "test\Video"

on_image(image_path, predtictor)
#on_video(videoPath, predtictor)