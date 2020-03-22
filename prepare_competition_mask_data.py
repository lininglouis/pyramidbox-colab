# -*- coding: utf-8 -*-
#
import glob
import xml.etree.ElementTree as ET
import os
from data.config_competition_mask import cfg
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import handy



def get_image_label_pair(COMPETITION_MASK_DATA_DIR):

    img_paths = glob.glob('{}/*.jpg'.format(COMPETITION_MASK_DATA_DIR))
    labels = glob.glob('{}/*.xml'.format(COMPETITION_MASK_DATA_DIR))
    img_paths.sort()
    labels.sort()

    effective_imagesPath = []
    effective_labelsPath = []
    for imgPath in tqdm(img_paths, total=len(img_paths)):
        corre_labelPath = imgPath.replace('jpg', 'xml')
        if os.path.exists(corre_labelPath) and os.path.exists(imgPath):
            effective_imagesPath.append(imgPath)
            effective_labelsPath.append(corre_labelPath)

    return effective_imagesPath, effective_labelsPath


def prepare_competition_MASK():
    img_paths, labels = get_image_label_pair(cfg.FACE.DATA_DIR)
    train_img_paths, val_img_paths, train_labels, val_labels = \
        train_test_split(img_paths, labels, test_size=0.20, random_state=42)

    for i in range(10):
        print(img_paths[i], labels[i])
    print('----------------------')

    for i in range(10):
        print(train_img_paths[-i], train_labels[-i])
    print('----------------------')

    for i in range(10):
        print(val_img_paths[-i], val_labels[-i])
    print('verification=-==============above==============')

    handy.generate_txt(train_img_paths, train_labels, cfg.FACE.TRAIN_FILE)
    handy.generate_txt(val_img_paths, val_labels, cfg.FACE.VAL_FILE)






def prepare_competition_MASK_for_classification():
    global COMPETITION_MASK_DATA_DIR
    img_paths = glob.glob('{}/*.jpg'.format(COMPETITION_MASK_DATA_DIR))
    labels = glob.glob('{}/*.xml'.format(COMPETITION_MASK_DATA_DIR))
    img_paths.sort()
    labels.sort()

    handy.mkdir_if_not_exists('./MASK_CLASSIFICATION_DATA')
    handy.mkdir_if_not_exists('./MASK_CLASSIFICATION_DATA/images')
    MASK_CLASSIFICATION_DATA_DIR = r'./mask_data/MASK_CLASSIFICATION_DATA/images'
    MASK_CLASSIFICATION_LABEL_PATH = r'./mask_data/MASK_CLASSIFICATION_DATA/label.txt'
    f_label = open(MASK_CLASSIFICATION_LABEL_PATH, 'w+')

    bbox = []

    '''
        0. mask 
        1. head 
        2. back 
        3. mid_mask 
    '''
    label_code_dict = {'mask': 0, 'head': 1, 'back': 2, 'mid_mask': 3}

    for img_path, label in zip(img_paths, labels):

        filename = os.path.basename(img_path)
        fprefix, ftype = filename.split('.')
        img = cv2.imread(img_path)
        filename, size, objects = read_xml(label)
        for idx, ob in enumerate(objects):
            class_code = label_code_dict[ob['label']]
            xmin, ymin, xmax, ymax = ob['boxes'][0]
            box_img = img[ymin:ymax + 1, xmin:xmax + 1]

            box_img_path = os.path.join(MASK_CLASSIFICATION_DATA_DIR, f'{fprefix}_{idx}.{ftype}')
            f_label.write(f'{box_img_path} {class_code}\n')
            cv2.imwrite(box_img_path, box_img)

    f_label.close()




if __name__ == '__main__':
    # settings. data_dir, train_txt_path, test_txt_path
    prepare_competition_MASK()
    #prepare_competition_MASK_for_classification()




