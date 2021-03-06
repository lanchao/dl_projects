import cv2
from shutil import copyfile

SRC_DIR = '/home/chaolan/data/chaolan/data/kitti/training/'
LABEL_DIR = 'label_2/'
TOTAL_IMG_NUM = 7481
TEST_START = 6800

flip_img = True
minsize = 25  # in px
object_class = 'car'  # car, pedestrian
size = 'large'  # small, large
difficulty = 'hard'  # easy, medium, hard

if object_class == 'car':
    #type_set = set(['Car', 'Van', 'Truck'])
    type_set = set(['Car'])
elif object_class == 'van_truck':
    #type_set = set(['Car', 'Van', 'Truck'])
    type_set = set(['Van', 'Truck'])
else:
    type_set = set(['Pedestrian'])

if size == 'large':
    scale = 1
else:
    scale = .5

if difficulty == 'easy':
    occlusion_set = set([0, 1])
    truncation_threshold = 0
elif difficulty == 'medium':
    occlusion_set = set([0, 1])
    truncation_threshold = 0.6
elif difficulty == 'hard':
    occlusion_set = set([0, 1, 2])
    truncation_threshold = 0.7
else:
    occlusion_set = set([0, 1, 2, 3])
    truncation_threshold = 1.0

flip = '_flip' if flip_img else ''
train_file = 'kitti_{:s}_train_{:s}_{:s}{:s}.idl'.format(object_class, difficulty, size, flip)
test_file = 'kitti_{:s}_test_{:s}_{:s}{:s}.idl'.format(object_class, difficulty, size, flip)
IMAGE_DIR = 'image_2_{:s}/'.format(size)
FLIP_DIR = 'image_2_{:s}/flip/'.format(size)
TESTING_DIR = 'image_2_testing/'
TESTING_FLIP_DIR = TESTING_DIR + 'flip/'
DEST_LABEL='/home/chaolan/data/chaolan/data/kitti/training/testing_label/'


def gen_testing_idl(data_range, test_base, file_name, flip_img):
    total_count = 0
    fout = open(SRC_DIR + file_name, 'w')
    for i in data_range:
        label_file = SRC_DIR + LABEL_DIR + '{:06d}.txt'.format(i)
        image_file = IMAGE_DIR + '{:06d}.png'.format(i)
        img = cv2.imread(SRC_DIR + image_file)
        out_image = TESTING_DIR + '{:06d}.png'.format(i - test_base)
        cv2.imwrite(SRC_DIR + out_image, img)
        if flip_img:
            flip_img = cv2.imread(FLIP_DIR + image_file)
            img_height, img_width = img.shape[:2]
            flip_image_out = TESTING_FLIP_DIR + '{:06d}.png'.format(i - test_base)
            cv2.imwrite(SRC_DIR + flip_image_out, flip_img)
        count = 0
        dest_label = DEST_LABEL + '{:06d}.txt'.format(i - test_base)
        copyfile(label_file, dest_label)
        with open(label_file, 'r') as f:
            is_first = True
            out1 = ''
            out2 = ''
            for line in f:
                comps = line.strip('\n').split(' ')
                object_type = comps[0]
                truncated = float(comps[1])
                occluded = int(comps[2])
                bbox = [float(x)*scale for x in comps[4:8]]
                if flip_img:
                    bbox2 = [img_width-bbox[2], bbox[1], img_width-bbox[0]-1, bbox[3]]
                if (object_type in type_set) and (occluded in occlusion_set) and \
                        (0 <= truncated <= truncation_threshold) and \
                        (bbox[2] - bbox[0] >= minsize) and \
                        (bbox[3] - bbox[1] >= minsize):
                    if is_first:
                        out1 += "\"" + out_image + "\": "
                        if flip_img:
                            out2 += "\"" + flip_image_out + "\": "
                        is_first = False
                    else:
                        out1 += ", "
                        if flip_img:
                            out2 += ", "
                    out1 += "({0:.1f}, {1:.1f}, {2:.1f}, {3:.1f})".format(bbox[0], bbox[1], bbox[2], bbox[3])
                    count += 1
                    total_count += 1
                    if flip_img:
                        out2 += "({0:.1f}, {1:.1f}, {2:.1f}, {3:.1f})".format(bbox2[0], bbox2[1], bbox2[2], bbox2[3])
                        count += 1
                        total_count += 1
        if count > 0:
            fout.write(out1 + ";\n")
            if flip_img:
                fout.write(out2 + ";\n")
    fout.close()
    print ("total_count: " + str(total_count))


gen_testing_idl(range(6800, 7481), 6800, 'benchmark_testing', True)
