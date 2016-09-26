import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc

from train import build_forward
from utils import googlenet_load
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from data_preprocessing.file_utils import create_dir

import cv2
import argparse
import numpy as np
import time
from datetime import datetime


def get_results(args, H):
    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    googlenet = googlenet_load.init(H)
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H,
                                                                                                        tf.expand_dims(
                                                                                                            x_in, 0),
                                                                                                        googlenet,
                                                                                                        'test',
                                                                                                        reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
                                      [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test',
                                                                  reuse=None)
    saver = tf.train.Saver()

#    total_image = 681
    total_image = args.total_img
    model_width = args.model_width
#    model_height = 384
    model_height = args.model_height

    SRC_DIR = args.src
    OUT_DIR = args.output
    RESULT_DIR = OUT_DIR + 'results/' + args.expname + '/data/'
    #IMG_DIR = SRC_DIR + 'image_2_testing/'
    IMG_DIR = SRC_DIR + 'image_2_large/'
    DUMP_DIR = OUT_DIR + 'results/' + args.expname + '/dump/'

    create_dir(RESULT_DIR)
    create_dir(DUMP_DIR)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.weights)

        for i in xrange(total_image):
            img_file = IMG_DIR + '{:06d}.png'.format(i)
            print 'Processing img ' + (IMG_DIR + '{:06d}.png'.format(i))
            img = cv2.imread(img_file)
            img_height = img.shape[0]
            img_width = img.shape[1]
            pad_height = model_height - img_height
            pad_width = model_width - img_width
            pad_img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            result_file = open(RESULT_DIR + '{:06d}.txt'.format(i), 'w')

            feed = {x_in: pad_img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.conf, tau=args.tau, show_removed=False)

            for rect in rects:
                if rect.score > args.conf:
                    output = 'Car -1 -1 -10 {0:.2f} {1:.2f} {2:.2f} {3:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {4:.2f}\n'.format(rect.x1, rect.y1, rect.x2, rect.y2, rect.score)
                    result_file.write(output)
            result_file.close()

            if args.dump:
                cv2.imwrite(DUMP_DIR + '{:06d}.png'.format(i), new_img)

def get_precision(result_file):
    recalls = []
    e_precisions = []
    m_precisions = []
    h_precisions = []
    with open(result_file) as f:
        for line in f:
            [recall, e_precison, m_precision, h_precision] = line.split(' ')
            recalls.append(float(recall))
            e_precisions.append(float(e_precison))
            m_precisions.append(float(m_precision))
            h_precisions.append(float(h_precision))

    e_sum = 0
    m_sum = 0
    h_sum = 0
    i = 0
    while i < len(e_precisions):
        e_sum += e_precisions[i]
        m_sum += m_precisions[i]
        h_sum += h_precisions[i]
        i += 4
    e_average = e_sum / 11
    m_average = m_sum / 11
    h_average = h_sum / 11

    return (e_average, m_average, h_average)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', required=True)
    #parser.add_argument('--exp_prefix', default=datetime.now().strftime('%Y_%m_%d.%H_%M'))
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--conf', default=0.0, type=float)
    parser.add_argument('--dump', default=False)
    parser.add_argument('--src', default='/home/chaolan/data/chaolan/data/kitti/training/')
    parser.add_argument('--output', default='/home/chaolan/kitti_benchmark/')
    parser.add_argument('--total_img', default=7481)
    parser.add_argument('--model_width', default=1248)
    parser.add_argument('--model_height', default=704)
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false')
    parser.add_argument('--precision', dest='precision', action='store_true')
    parser.add_argument('--no-precision', dest='precision', action='store_false')
    parser.set_defaults(evaluate=True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)

    if args.evaluate:
        get_results(args, H)

    if args.precision:
        result_path = args.output + 'results/' + args.expname + '/plot/car_detection.txt'
        precision_file = args.output + 'results/' + args.expname + '/plot/precision.txt'
        with open(precision_file, 'w') as f:
            precision = get_precision(result_path)
            f.write(str(precision))
            print 'Precision:'
            print precision

if __name__ == '__main__':
    main()
