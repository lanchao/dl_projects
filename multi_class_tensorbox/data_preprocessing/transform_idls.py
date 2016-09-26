import cv2

SRC_DIR = '/home/chaolan/data/chaolan/data/kitti/training/'
LABEL_DIR = 'label_2/'
TOTAL_IMG_NUM = 7481

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


def gen_idl(data_range, file_name):
    total_count = 0
    fout = open(SRC_DIR + file_name, 'w')
    for i in data_range:
        label_file = SRC_DIR + LABEL_DIR + '{:06d}.txt'.format(i)
        image_file = IMAGE_DIR + '{:06d}.png'.format(i)
        if flip_img:
            img = cv2.imread(SRC_DIR + image_file)
            img_height, img_width = img.shape[:2]
            flip_image_file = FLIP_DIR + '{:06d}.png'.format(i)
        count = 0
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
                        out1 += "\"" + image_file + "\": "
                        if flip_img:
                            out2 += "\"" + flip_image_file + "\": "
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


gen_idl(range(0, 6800), train_file)
gen_idl(range(6800, 7481), test_file)
