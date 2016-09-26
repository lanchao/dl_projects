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

import cv2
import argparse
import numpy as np
import time


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
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])),
                                      [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test',
                                                                  reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.weights)

        display_width = 1280

        video_capture = cv2.VideoCapture(args.video)

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        width = size[0]
        height = size[1]

        scale_ratio = display_width * 1.0 / width

        print 'original video: {:d} FPS && size: {:d} x {:d}'.format(int(fps), width, height)

        f_idx = 1
        speed = int(fps)
        # frame steps for calculating FPS
        speed_cal_step = 100
        start_time = time.time()

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            # record start time for FPS calculation
            if f_idx % speed_cal_step == 0:
                start_time = time.time()

            #img = imresize(frame, (H["image_height"], H["image_width"]), interp='cubic')
            img = cv2.copyMakeBorder(frame, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            #(np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.conf, tau=args.tau, show_removed=False)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(new_img, 'Frame speed={0:.2f}. Conf={1:.2f}'.format(speed, args.conf), (20, 30), font, 0.6, (255, 0, 0), 2,
                        cv2.LINE_AA)

            if args.ui:
                cv2.imshow('Object Detection'.format(speed), new_img)
            if f_idx % speed_cal_step == speed_cal_step - 1:
                dur = time.time() - start_time
                speed = int(1 * speed_cal_step / dur)

            if f_idx % speed_cal_step == 0:
                print 'frame: {:d}, speed: {:d} FPS'.format(f_idx, speed)

            if args.dump:

                cv2.imwrite(args.dump_path + str(f_idx) + '.png', new_img)

            f_idx += 1
            # press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--video', required=True)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--ui', default=True)
    parser.add_argument('--conf', default=0.5, type=float)
    parser.add_argument('--dump', default=False)
    parser.add_argument('--dump_path', default='/tmp/test_detector_out/')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)

    get_results(args, H)

if __name__ == '__main__':
    main()
