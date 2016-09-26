import os
import cv2

def check_img_size(SRC_DIR, height, width):
    for img_file in os.listdir(SRC_DIR):
        if img_file.endswith('.png'):
            print 'checking %s'%img_file
            img = cv2.imread(SRC_DIR+img_file)
            img_height = img.shape[0]
            img_width = img.shape[1]
            if img_height != height or img_width != width:
                print 'image file %s has incorrect size.'%img_file
                break
    print 'All clear!'

def find_img_by_size(SRC_DIR, height, width):
    for img_file in os.listdir(SRC_DIR):
        if img_file.endswith('.png'):
            print 'checking %s' % img_file
            img = cv2.imread(SRC_DIR + img_file)
            img_height = img.shape[0]
            img_width = img.shape[1]
            if img_height == height and img_width == width:
                print 'image file %s has right size.'%img_file
                break

#check_img_size('/home/chaolan/training_data/ucar_1248_for_kitti_car/2016_09_12.22_06/', 704, 1248)
find_img_by_size('/home/chaolan/data/training_data/', 1080, 1920)