import tensorflow as tf
import json

from train import build_forward
from utils import googlenet_load

import argparse

# 1. dump model using this script
# 2. freeze graph
#bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=../tensorbox/data/car_tensorbox.pb --input_checkpoint=../tensorbox/data/car.checkpoint.data --output_graph=/tmp/car_frozen_graph.pb --input_binary=True --output_node_names=add,Reshape_2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', required=True)
    parser.add_argument('--model', required=True)   # checkpoint file during training
    parser.add_argument('--out_graph', required=True)
    parser.add_argument('--out_checkpoint', required=True)
    args = parser.parse_args()

    with open(args.hypes, 'r') as f:
        H = json.load(f)

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    googlenet = googlenet_load.init(H)
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = \
                build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, \
                [grid_area * H['rnn_len'], H['num_classes']])), [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = \
                build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, ".", args.out_graph, False)
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.model)
        saver.save(sess, args.out_checkpoint)
    print pred_boxes, pred_confidences

if __name__ == '__main__':
    main()
