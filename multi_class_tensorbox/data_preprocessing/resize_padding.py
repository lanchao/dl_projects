# resize the image and maintain the aspect ratio. pad the image if necessary
from __future__ import print_function
import cv2
import numpy as np
import os
import errno

SRC_DIR = '/home/chaolan/data/chaolan/data/kitti/training/image_2/'
TOTAL_IMG_NUM = 2148

flip = True
FLIP_DIR = SRC_DIR + 'flip/'
target = 'large'

if flip:
    if not os.path.exists(os.path.dirname(FLIP_DIR)):
        try:
            os.makedirs(os.path.dirname(FLIP_DIR))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if target == 'small':
    DEST_DIR = '/home/chaolan/data/chaolan/data/kitti/training/small/'
    scale = 0.9
    target_width = 640
    target_height = 480
else:
    DEST_DIR = '/home/chaolan/data/chaolan/data/kitti/training/large/'
    scale = 1
    target_width = 1248
    target_height = 384

for img_file in os.listdir(SRC_DIR):
    if img_file.endswith('.png'):
        img = cv2.imread(SRC_DIR+img_file)
        if scale != 1:
            resize_img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resize_img = img
        target_image = np.zeros((target_height, target_width, 3), np.uint8)
        cutoff_width = target_width if (resize_img.shape[1] > target_width) else resize_img.shape[1]
        cutoff_height = target_height if (resize_img.shape[0] > target_height) else resize_img.shape[0]
        target_image[:cutoff_height, :cutoff_width, :] = resize_img[:cutoff_height, :cutoff_width, :]
        cv2.imwrite(DEST_DIR + img_file, target_image)
        if flip:
            flip_img = cv2.flip(target_image, 1)
            cv2.imwrite(FLIP_DIR + img_file, flip_img)
        print(img_file+'\n')
    print('\ndone\n')
