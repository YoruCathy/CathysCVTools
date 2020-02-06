import cv2 as cv
import json
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from time import time
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvbase as cvb

root = "DATASET_PATH"
train_json_path = root+"/annotations/"+"instances_train.json"

COCO_CLASSES = coco_classes or your own dataset classes here

COCO_LABEL_MAP = {1: 1, 2: 2, 4: 3, 5: 4,
                  6: 5, 7: 6, 8: 7}


def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)
    for img in imgs:
        image_id = img["id"]
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


def get_bounding_box(mask):
    coords = np.transpose(np.nonzero(mask))
    y, x, h, w = cv.boundingRect(coords)
    return x, y, w, h


def run_one_image(img_id, annos, img_name):
    img_height =
    img_width = 
    img_info_dict = []
    rle_list = []

    for anno in annos:
        # print(anno)
        objects_info = {}
        x, y, w, h = anno['bbox']
        cat_id = anno['category_id']
        mask = maskUtils.decode(annToRLE(anno, img_height, img_width))
        xx, yy, ww, hh = get_bounding_box(mask)
        instance_mask_ = mask[yy:yy+hh, xx:xx+ww].astype(np.bool) * 255
        instance_mask_ = Image.fromarray(instance_mask_.astype(
            np.uint8)).resize((64, 64), Image.NEAREST)
        instance_mask_ = np.reshape(instance_mask_, (-1, 64 * 64))
        rle = maskUtils.encode(instance_mask_)

        # Convert from 1-90 to 1-80
        objects_info['label'] = COCO_LABEL_MAP[cat_id]
        objects_info['bbox'] = (x, y, w, h)  # TO BE CAREFUL
        objects_info['img_wh'] = (img_width, img_height)
        objects_info['inst_id'] = anno['id']
        rle_list.append(rle)
        img_info_dict.append(objects_info)
    info_txt = np.zeros((len(img_info_dict), 9 + num_bases))
    for i in range(len(img_info_dict)):
        info_txt[i][0] = img_info_dict[i]['label']
        info_txt[i][1:3] = img_info_dict[i]['img_wh']
        info_txt[i][3:7] = img_info_dict[i]['bbox']
        info_txt[i][7] = img_info_dict[i]['inst_id']
    img_info = np.reshape(info_txt, (-1, 9))
    cat_list = []  # Cat list in one img
    for i in range(len(img_info)):
        cat_id = int(img_info[i][0])
        cat_list.append(cat_id)

    save_xml(str(img_name.split('.')[0]).zfill(12), cat_list, img_info, rle_list,
             save_dir, img_width, img_height, 3)


def annToRLE(ann, h, w):
    """
    Thanks to pycocotools
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def save_xml(img_name, cat_list, points_list, rle_list, save_dir, width, height, channel):
    has_objects = False
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "JPEGImages"

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
    count = 0
    for points, rle in zip(points_list, rle_list):
        bbox_xmin, bbox_ymin = points[3], points[4]
        bbox_w, bbox_h = points[5], points[6]
        bbox_xmax, bbox_ymax = bbox_xmin + bbox_w, bbox_ymin+bbox_h
        # if bbox_xmin < 0:
        #     bbox_xmin = 0
        assert(bbox_xmin < bbox_xmax)
        assert(bbox_ymin < bbox_ymax)
        if bbox_w <= 0:
            print(bbox_w)
        if bbox_h <= 0:
            print(bbox_h)
        inst_id = points[7]

        coef_str = str(points[9:])
        coef_str = coef_str[1:-1]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        # print(cat_list[count])
        node_name.text = COCO_CLASSES[cat_list[count]-1]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % bbox_xmin
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % bbox_ymin
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % bbox_xmax
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bbox_ymax
        node_polygon = SubElement(node_object, 'rle')
        node_polygon.text = '%s' % rle
        node_inst_id = SubElement(node_object, 'inst_id')
        node_inst_id.text = '%s' % int(inst_id)
        count += 1
        has_objects = True
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    save_xml = os.path.join(save_dir, img_name + '.xml')
    with open(save_xml, 'wb') as f:
        f.write(xml)


if __name__ == "__main__":
    print('Loading', os.path.join(root, 'annotations', 'instances_train.json'))
    anns = cvb.load(train_json_path)
    # print(anns)
    imgs_dict, anns_dict = make_json_dict(anns["images"], anns["annotations"])
    count = 0
    save_dir = os.path.join(root, '')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for img_id in tqdm(anns_dict.keys()):
        img_name = imgs_dict[img_id]
        # print(img_name)
        anns = anns_dict[img_id]
        # print(anns)
        run_one_image(img_id, anns, img_name)
        count += 1
