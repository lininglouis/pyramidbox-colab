from tqdm import tqdm
import xml
import xml.etree.ElementTree as ET
import os

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


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)



def generate_txt(img_paths, labels, filepath):
    bbox = []
    img_size = []
    print('reading label started--------')
    for label in tqdm(labels, total=len(labels)):
        filename, size, objects = read_xml(label)
        img_size.append(size)
        one_image_bboxes = [ob['boxes'][0] for ob in objects]
        bbox.append(one_image_bboxes)
    print('reading label finished----------')


    fw = open(filepath, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        im_height, im_width = img_size[index]
        boxes = bbox[index]
        fw.write('{}:{}'.format(path, len(boxes)))
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = cap_box(xmin, ymin, xmax, ymax, im_width,  im_height)
            if xmax > im_width or (ymax>im_height):
                print('break!!!==========', box, (height, width), im_height, im_width, path)
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