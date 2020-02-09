import pickle
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import os
from PIL import Image
from time import time
from sklearn.metrics import jaccard_similarity_score, jaccard_score
import sys

import warnings
from tqdm import tqdm


class Config:
    def __init__(self):
        self.dataset = 'sbd'
        self.path = '/home/data/ese/VOC/norm_mask_sbd_val'
        self.val_path = '/home/data/ese/VOC/norm_mask_sbd_train_no_val'
        self.scale = (128, 128)
        self.n_components = 50
        # self.n_components = list(range(10, 200, 10))
        # self.n_iter = [1, 10, 50, 100]
        self.n_iter = [1]
        self.auto_theta = False
        self.theta_list = range(114, 144, 1)  # No use
        self.print_extra_info = False
        self.dict = self.get_cat_dict(self.dataset)
        self.save = True
        self.save_path = f'sparse_{self.dataset}_{self.scale[0]}_{self.scale[1]}'
        # Create the folder
        if self.save:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.allow_save = True

        self.select_cat = None
        if self.select_cat is None:
            self.select_cat = self.dict.keys()

        if type(self.n_components) == type(1):
            self.n_components = [self.n_components]
        if type(self.n_iter) == type(1):
            self.n_iter = [self.n_iter]

    def set_start_end(self, start, end):
        # include the end!!!!
        self.select_cat = list(range(start, end+1))

    def get_cat_dict(self, dataset):
        coco_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                     9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                     16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                     24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                     34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                     40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                     46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                     53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
                     60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                     70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                     78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                     86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

        sbd_dict = {1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat",
                    9: "chair", 10: "cow", 11: "dining table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
                    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "monitor"}

        sbd_coco_dict = {1: "airplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat",
                         9: "chair", 10: "cow", 11: "dining table", 12: "dog", 13: "horse", 14: "motorcycle", 15: "person",
                         16: "potted plant", 17: "sheep", 18: "couch", 19: "train", 20: "tv"}

        if(dataset == 'coco'):
            return coco_dict
        elif (dataset == 'sbd'):
            return sbd_dict
        else:
            raise Exception('dataset must be either coco or sbd')
    # dict (voc_id:coco_id)
    # voc_to_coco_dict = {1:5, 2:2, 3:16, 4:9, 5:44, 6:6, 7:3, 8:17, 9:62, 10:21, 11:67, 12:18, 13:19, 14:4, 15:1,
    #                  16:64, 17:20, 18:63, 19:7, 20:72}


class Logger:
    def __init__(self):
        self.storage = []

    def add(self, id, n_comp, n_iter, mIOU, extra_info=''):
        self.storage.append({'id': id, 'n_comp': n_comp,
                             'n_iter': n_iter, 'mIOU': mIOU, 'best_IOU': extra_info})

    def print(self):
        print('|'.join(list(self.storage[0].keys())))
        for info in self.storage:
            first = True
            for _, value in info.items():
                if first:
                    print(f'{value}', end='')
                    first = False
                else:
                    print(f'|{value}', end='')
            print('\n', end='')

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('|'.join(list(self.storage[0].keys())) + '\n')
            for info in self.storage:
                first = True
                for _, value in info.items():
                    if first:
                        f.write(f'{value}')
                        first = False
                    else:
                        f.write(f'|{value}')
                f.write('\n')


class Timer:
    def __init__(self):
        self.time = time()

    def stop(self, print_=True, refresh=True, start='', end='\n'):
        delta_time = time() - self.time
        if print_:
            print(start + "%.2f seconds used" %
                  (time() - self.time), end=end)
        if refresh:
            self.time = time()
        return delta_time


def readCategory(path, cat_dict, cat_id, scale, normalize):
    path += "_" + str(scale[0])
    # voc_to_coco_dict = {1:5, 2:2, 3:16, 4:9, 5:44, 6:6, 7:3, 8:17, 9:62, 10:21, 11:67, 12:18, 13:19, 14:4, 15:1,
    #                  16:64, 17:20, 18:63, 19:7, 20:72}
    for i in range(1, 21):
        pass
    mask_list = os.listdir(path+"/"+cat_dict[cat_id])
    all_masks = np.zeros((len(mask_list), scale[0] * scale[1]))
    for i in tqdm(range(len(mask_list))):
        mask = np.array(Image.open(
            # path + "/" + cat_dict_coco[voc_to_coco_dict[cat_id]] + "/" + mask_list[i]))
            path + "/" + cat_dict[cat_id] + "/" + mask_list[i]))
        all_masks[i] = mask.flatten()
    if normalize:
        all_masks = (all_masks - all_masks.mean(axis=0)) / \
            all_masks.std(axis=0)
    return all_masks


def readAllMasks(path, scale, normalize):
    tmp = Timer()
    path += "_" + str(scale[0])
    # mask_list = os.listdir(path + "/" + cat_dict[cat_id])

    mask_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            mask_list.append(os.path.join(r, file))

    all_masks = np.zeros((len(mask_list), scale[0] * scale[1]))
    for i, img in enumerate(tqdm(mask_list)):
        mask = np.array(Image.open(img))
        all_masks[i] = mask.flatten()
    if normalize:
        all_masks = (all_masks - all_masks.mean(axis=0)) / \
            all_masks.std(axis=0)

    tmp.stop()
    return all_masks


def scale_to_255(bases):
    new = (bases - bases.min()) / (bases.max() - bases.min()) * 255
    return new.astype('uint8')


def binarize(patches, threshold=128):
    return (patches > threshold) * 255


if __name__ == '__main__':
    cfg = Config()
    log = Logger()
    timer = Timer()

    # Start and end
    if len(sys.argv) > 1:
        cfg.set_start_end(int(sys.argv[1]), int(sys.argv[2]))

    images = readAllMasks(cfg.path, cfg.scale, False)
    # print(f"train images {images.shape[0]}")
    timer.stop(print_=False)

    for n_iter in cfg.n_iter:
        for n_components in cfg.n_components:
            print(
                f'Start training dictionary with {n_components} bases and {n_iter} max iters')
            dico = MiniBatchDictionaryLearning(
                n_components=n_components, n_jobs=-1, n_iter=n_iter)
            hit = False
            if cfg.allow_save and os.path.exists(f'{cfg.save_path}/all_{n_components}_{n_iter}.sklearnmodel'):
                dico = pickle.load(
                    open(f'{cfg.save_path}/all_{n_components}_{n_iter}.sklearnmodel', 'rb'))
                print(
                    f'Use hitted {cfg.save_path}/all_{n_components}_{n_iter}.sklearnmodel')
                hit = True
            else:
                dico = DictionaryLearning(
                    n_components=n_components, n_jobs=-3, max_iter=n_iter, verbose=True)
                dico.fit(images)
                print(f'{dico.n_iter_} iters')
            timer.stop(start=' ')
            n_iter_actual = dico.n_iter_

            if cfg.save and not hit:
                np.save(
                    f'{cfg.save_path}/all_{n_components}_{n_iter_actual}', dico.components_)
                pickle.dump(dico, open(
                    f'{cfg.save_path}/all_{n_components}_{n_iter_actual}.sklearnmodel', 'wb'))

            # Calculate the mIOU based on cats
            for cat_id in cfg.select_cat:
                images_val = readCategory(
                    cfg.val_path, cfg.dict, cat_id, cfg.scale, normalize=False)

                # Calculating the coeffs
                coeffs = dico.transform(images_val)  # Use val here
                # coeffs = np.clip(coeffs, -2500, 2500)
                patches = np.dot(coeffs, dico.components_)
                # print(np.unique(patches))
                # patches_255 = scale_to_255(patches)
                # patches = patches >= 0  # TODO

                mIOU, mIOU_a, mIOU_m = 0, 0, 0
                mIOU_best = 0
                theta_record = {i: 0 for i in cfg.theta_list}
                result = []
                count = 0
                for i, patch in enumerate(tqdm(patches)):
                    # IOU
                    # mIOU += jaccard_score(images_val[i].astype(
                    #     np.uint8)/255, binarize(patches[i]).astype(np.uint8)/255)
                    # TODO
                    # mIOU += jaccard_score(images_val[i].astype(
                    #     np.uint8), patches[i].astype(np.uint8))
                    mIOU_list = []
                    if cfg.auto_theta:
                        for theta in cfg.theta_list:
                            temp_patch = patch >= theta
                            mIOU_list.append(jaccard_score(images_val[i].astype(
                                np.uint8)/255, temp_patch.astype(np.uint8)))
                            # log.add(cat_id, n_components, n_iter_actual, mIOU /
                            #         (patches.shape[0] + 1), mIOU_best / (patches.shape[0] + 1), theta)
                        # mIOU_best += max(mIOU_list)
                        # print(index * 10 + 10)
                        for theta in cfg.theta_list:
                            result.append(0)
                        c = 0
                        for theta in cfg.theta_list:
                            result[c] += mIOU_list[c]
                            c += 1
                    else:
                        max, min = np.unique(
                            patch)[-1], np.unique(patch)[0]
                        average = (max+min)/2
                        temp_patch_a = patch >= average
                        mIOU_a += jaccard_score(images_val[i].astype(
                            np.uint8)/255, temp_patch_a.astype(np.uint8))
                        mean = patch.mean()
                        temp_patch_m = patch >= mean
                        j = jaccard_score(images_val[i].astype(
                            np.uint8)/255, temp_patch_m.astype(np.uint8))
                        mIOU += j
                        if j >= 0.8:
                            count += 1
                # print(theta_record)
                # Record the result
                log.add(cat_id, count/patches.shape[0], n_iter_actual, mIOU_m /
                        (patches.shape[0]), mIOU_a / (patches.shape[0]))
                log.save('sbd_all_mean2_single.txt')
