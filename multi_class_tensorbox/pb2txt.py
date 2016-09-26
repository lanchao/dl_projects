import tensorflow as tf
import os
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='pb2txt')
    parser.add_argument('--model_path', dest='model_path',
                        help='model path',
                        default='/home/yfzhong/data/projects/demo/models/ucar_mc_0830_small.pb', type=str)
    parser.add_argument('--dest_dir', dest='dest_dir',
                        help='destination directory',
                        default='./', type=str)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_path = args.model_path

    graph = tf.Graph()
    graph_def = tf.GraphDef()
    tf.set_random_seed(0)
    with open(model_path) as f:
        tf.set_random_seed(0)
        graph_def.MergeFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    with tf.Session():
        tf.train.write_graph(graph_def, './', '{:s}.txt'.format(model_path.split('/')[-1]), True)

    print os.path.join(args.dest_dir, '{:s}.txt'.format(model_path.split('/')[-1]))

if __name__ == '__main__':
    main()