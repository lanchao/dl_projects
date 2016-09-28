import tensorflow as tf
import os
import numpy as np
import tensorflow.contrib.slim as slim
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

def load_net(H, inputs, reuse):

    arg_scope = inception_resnet_v2_arg_scope()

    with slim.arg_scope(arg_scope):
        endpoints = inception_resnet_v2(inputs=inputs, num_classes=1001, is_training=False, reuse=reuse)

        coarse_feat = endpoints['Conv2d_7b_1x1']
        early_feat = endpoints['Mixed_5b']
        early_feat_channels = 320

        print 'cnn_shape'
        print coarse_feat.get_shape()

        return coarse_feat, early_feat, early_feat_channels