# -*- coding: utf-8 -*-
#
import glob
import xml.etree.ElementTree as ET
import os
from data.config_competition_mask import cfg
import cv2
from sklearn.model_selection import train_test_split

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


def get_image_label_pair(COMPETITION_MASK_DATA_DIR):

    img_paths = glob.glob('{}/*.jpg'.format(COMPETITION_MASK_DATA_DIR))
    labels = glob.glob('{}/*.xml'.format(COMPETITION_MASK_DATA_DIR))
    img_paths.sort()
    labels.sort()

    effective_imagesPath = []
    effective_labelsPath = []
    for imgPath in img_paths:
        corre_labelPath = imgPath.replace('jpg', 'xml')
        if os.path.exists(corre_labelPath) and os.path.exists(imgPath):
            effective_imagesPath.append(imgPath)
            effective_labelsPath.append(corre_labelPath)

    return effective_imagesPath, effective_labelsPath


def prepare_competition_MASK(COMPETITION_MASK_DATA_DIR):
    img_paths, labels = get_image_label_pair(COMPETITION_MASK_DATA_DIR)
    MASK_TRAIN_FILE = cfg.FACE.TRAIN_FILE
    MASK_VAL_FILE = cfg.FACE.VAL_FILE
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

    generate_txt(train_img_paths, train_labels, MASK_TRAIN_FILE)
    generate_txt(val_img_paths, val_labels, MASK_VAL_FILE)



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
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax > im_width or (ymax>im_height):
                print('break!!!==========', box, (height, width), cv2.imread(path).shape[:2], path)
                print(img_paths[index], labels[index])

            width = (xmax - xmin) + 1
            height = (ymax - ymin) + 1
            data = ' {} {} {} {} {}'.format(xmin, ymin, width, height, 1)
            fw.write(data)
        fw.write('\n')
    fw.close()


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def prepare_competition_MASK_for_classification():
    global COMPETITION_MASK_DATA_DIR
    img_paths = glob.glob('{}/*.jpg'.format(COMPETITION_MASK_DATA_DIR))
    labels = glob.glob('{}/*.xml'.format(COMPETITION_MASK_DATA_DIR))
    img_paths.sort()
    labels.sort()

    mkdir_if_not_exists('./MASK_CLASSIFICATION_DATA')
    mkdir_if_not_exists('./MASK_CLASSIFICATION_DATA/images')
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
    COMPETITION_MASK_DATA_DIR = r'./mask_data/10'
    #COMPETITION_MASK_DATA_DIR = '/home/data/10'
    prepare_competition_MASK(COMPETITION_MASK_DATA_DIR)


    #prepare_competition_MASK_for_classification()




