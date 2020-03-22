# -*- coding: utf-8 -*-
#
import glob
import xml.etree.ElementTree as ET
import os
from data.config_open_mask import cfg
import cv2
from sklearn.model_selection import train_test_split

import glob
import os
from pathlib import Path
from tqdm import tqdm
from utils import handy


def get_clean_data_pairs(imagePaths):
  effective_imagesPath = []
  effective_labelsPath = []
  for imgPath in tqdm(imagePaths, total = len(imagePaths)):
    imgPath = Path(imgPath)
    corre_labelPath = imgPath.parent.parent.joinpath('label').joinpath(imgPath.stem+'.xml')
    if os.path.exists(corre_labelPath) :
        effective_imagesPath.append(str(imgPath))
        effective_labelsPath.append(str(corre_labelPath))
  return effective_imagesPath, effective_labelsPath



def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

if __name__ == '__main__':
    #prepare_competition_MASK(cfg.FACE.DATA_DIR)
    #prepare_competition_MASK_for_classification()

    labelPaths = glob.glob(os.path.join(cfg.FACE.DATA_DIR, "label", '*.xml'))
    imagePaths = glob.glob(os.path.join(cfg.FACE.DATA_DIR, "data", '*.jpg'))
    imagePaths, labelPaths = get_clean_data_pairs(imagePaths[:])
    train_img_paths, val_img_paths, train_labels, val_labels = \
        train_test_split(imagePaths, labelPaths, test_size=0.20, random_state=42)


    print('start generatiing. train ....')
    handy.generate_txt(train_img_paths, train_labels, cfg.FACE.TRAIN_FILE)
    print('start generatiing.  val....')
    handy.generate_txt(val_img_paths, val_labels, cfg.FACE.VAL_FILE)





