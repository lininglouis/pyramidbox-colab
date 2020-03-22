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



def get_clean_data_pairs(imagePaths):
  effective_imagesPath = []
  effective_labelsPath = []
  for imgPath in imagePaths:
    imgPath = Path(imgPath)
    corre_labelPath = imgPath.parent.parent.joinpath('label').joinpath(imgPath.stem+'.xml')
    if os.path.exists(corre_labelPath) :
        effective_imagesPath.append(str(imgPath))
        effective_labelsPath.append(str(corre_labelPath))
  return effective_imagesPath, effective_labelsPath

def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size_elem = root.find('size')
    height = int(size_elem.find('height').text)
    width = int(size_elem.find('width').text)

    objects = []
    for object_elem in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        obj = {}
        obj['label'] = object_elem.find('name').text
        list_with_all_boxes = []
        for box in object_elem.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
        one_box = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(one_box)
        obj['boxes'] = list_with_all_boxes
        objects.append(obj)

    return filename, (height, width), objects


def generate_txt(img_paths, labels, filepath):
    bbox = []
    for label in labels:
        filename, size, objects = read_xml(label)
        one_image_bboxes = [ob['boxes'][0] for ob in objects]
        bbox.append(one_image_bboxes)

    fw = open(filepath, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        im_height, im_width = cv2.imread(path).shape[:2]
        boxes = bbox[index]
        fw.write('{}:{}'.format(path, len(boxes)))
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax > im_width or (ymax>im_height):
                print('break!!!==========', box, (height, width), cv2.imread(path).shape[:2], path)
                print(img_paths[index], labels[index])

            width = (xmax - xmin) + 1
            height = (ymax - ymin) + 1
            data = ':{}:{}:{}:{}:{}'.format(xmin, ymin, width, height, 1)
            fw.write(data)
        fw.write('\n')
    fw.close()




def cap_box(xmin, ymin, xmax, ymax, im_width, im_height):
    '''
        Cap box info within range of 0 and large boundary
        we assume box follows the order of
            xmin, ymin, xmax, ymax
    '''
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(im_width-1, xmax)
    ymax = min(im_height-1, ymax)
    return xmin, ymin, xmax, ymax

def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

if __name__ == '__main__':
    #prepare_competition_MASK(cfg.FACE.DATA_DIR)
    #prepare_competition_MASK_for_classification()

    labelPaths = glob.glob(os.path.join(cfg.FACE.DATA_DIR, "label", '*.xml'))
    imagePaths = glob.glob(os.path.join(cfg.FACE.DATA_DIR, "data", '*.jpg'))
    imagePaths, labelPaths = get_clean_data_pairs(imagePaths)
    train_img_paths, val_img_paths, train_labels, val_labels = \
        train_test_split(imagePaths, labelPaths, test_size=0.20, random_state=42)


    generate_txt(train_img_paths, train_labels, cfg.FACE.TRAIN_FILE)
    generate_txt(val_img_paths, val_labels, cfg.FACE.VAL_FILE)





