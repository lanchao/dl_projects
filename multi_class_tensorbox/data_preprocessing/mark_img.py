import os
import errno
import cv2

def mark(idl_path):
    base_name, image_name = os.path.split(idl_path)
    base_dir = base_name + '/'
    with open(idl_path) as idl:
        for line in idl:
            line = line.strip()
            if line.startswith('\"'):
                rects = []
                line = line[:-1]
                data = line.split(':')
                filename = data[0].strip("\"")
                input_image_path = base_dir + filename
                dir_name, image_name = os.path.split(input_image_path)
                marked_dir = dir_name + '/marked/'
                if not os.path.exists(os.path.dirname(marked_dir)):
                    try:
                        os.makedirs(os.path.dirname(marked_dir))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                print line
                start = True
                if len(data) > 1:
                    annos = data[1].split('),')
                    for anno in annos:
                        anno = anno.strip()
                        if anno.endswith(')'):
                            anno = anno[:-1]
                        if anno.startswith('('):
                            anno = anno[1:]
                        cords = anno.split(', ')
                        print cords
                        x_m = ceiling(float(cords[0].strip()))
                        y_m = ceiling(float(cords[1].strip()))
                        x_l = ceiling(float(cords[2].strip()))
                        y_l = ceiling(float(cords[3].strip()))
                        rect = ((x_m, y_m) , (x_l,y_l))
                        rects.append(rect)
                    add_rect(marked_dir, input_image_path, rects)

def mark_idl_with_addx(idl_path):
    base_name, image_name = os.path.split(idl_path)
    base_dir = base_name + '/'
    with open(idl_path) as idl:
        for line in idl:
            line = line.strip()
            if line.startswith('\"'):
                rects = []
                line = line[:-1]
                data = line.split('\":')
                filename = data[0].strip("\"")
                input_image_path = base_dir + filename
                dir_name, image_name = os.path.split(input_image_path)
                marked_dir = dir_name + '/marked/'
                if not os.path.exists(os.path.dirname(marked_dir)):
                    try:
                        os.makedirs(os.path.dirname(marked_dir))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                print line
                start = True
                if len(data) > 1:
                    annos = data[1].split('(')[1:]
                    for anno in annos:
                        cords_anno = anno.split(':')[0]
                        if cords_anno.endswith(')'):
                            cords_anno = cords_anno[:-1]
                        if cords_anno.startswith('('):
                            cords_anno = cords_anno[1:]
                        cords = cords_anno.split(',')
                        print cords
                        x_m = ceiling(float(cords[0].strip()))
                        y_m = ceiling(float(cords[1].strip()))
                        x_l = ceiling(float(cords[2].strip()))
                        y_l = ceiling(float(cords[3].strip()))
                        rect = ((x_m, y_m) , (x_l,y_l))
                        rects.append(rect)
                    add_rect(marked_dir, input_image_path, rects)


def add_rect(output_dir, input_image, rects):
    img = cv2.imread(input_image)
    for rect in rects:
        cv2.rectangle(img, rect[0], rect[1], (255,0,0), 1)
    filename = os.path.basename(input_image)

    if not os.path.exists(os.path.dirname(output_dir)):
        try:
            os.makedirs(os.path.dirname(output_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    cv2.imwrite(output_dir + filename, img)

def ceiling(x):
    n = int(x)
    return n if n-1 < x <= n else n+1

idl_path = '/home/chaolan/data/chaolan/data/UCAR/training_data/ucar_1248_for_kitti_car/2016_09_12.22_06.test.idl'

#mark(idl_path)
mark_idl_with_addx(idl_path)