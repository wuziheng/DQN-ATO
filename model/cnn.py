#!/usr/bin/env python
# encoding: utf-8
"""
@File   : game
@author : wuziheng
@Date   : 3/20/18
@license:
"""

import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from config import Config
import math


CONV_WEIGHT_DECAY = 0
FC_WEIGHT_DECAY = 0
FC_WEIGHT_STDDEV = 0.01

tf.app.flags.DEFINE_integer('input_size', 80, "input image size")
activation = tf.nn.relu


def mlp(x, num_classes=1):
    c = Config()
    # c['use_bias'] = True
    # c['padding'] = 'SAME'
    #
    # with tf.variable_scope('scale1'):
    #     c['conv_filters_out'] = 16
    #     c['ksize'] = 3
    #     c['stride'] = 1
    #     x = conv(x, c)
    #     x = activation(x)
    #     print "x1_shape", x.get_shape()
    #
    # with tf.variable_scope('scale2'):
    #     x = _max_pool(x, ksize=2, stride=2)
    #     c['conv_filters_out'] = 32
    #     c['ksize'] = 3
    #     c['stride'] = 1
    #     x = conv(x, c)
    #     x = activation(x)
    #     print "x2_shape", x.get_shape()
    #
    # x = tf.reshape(x,[-1,feature_num*feature_num*32])
    with tf.variable_scope('fc1'):
        c['fc_units_out'] = 24
        x = fc(x, c)

    with tf.variable_scope('fc_out'):
        c['fc_units_out'] = num_classes
        x = fc(x, c)
    return x

def dueling(x, num_classes=5,
              scope_name='target_net',
              use_bias=True,  # defaults to using batch norm
             ):
    c = Config()
    c['use_bias'] = use_bias
    c['padding'] = 'SAME'

    with tf.variable_scope(scope_name):
        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 32
            c['ksize'] = 8
            c['stride'] = 4
            x = conv(x, c)
            x = activation(x)
            print "x1_shape", x.get_shape()

        with tf.variable_scope('scale2'):
            # x = _max_pool(x, ksize=2, stride=2)
            c['conv_filters_out'] = 32
            c['ksize'] = 4
            c['stride'] = 2
            x = conv(x, c)
            x = activation(x)
            print "x2_shape", x.get_shape()

        with tf.variable_scope('scale3'):
            c['conv_filters_out'] = 64
            c['ksize'] = 3
            c['stride'] = 1
            x = conv(x, c)
            x = activation(x)
            print "x3_shape", x.get_shape()


        x = tf.reshape(x, [-1, 6400])
        #
        with tf.variable_scope('fc1'):
            c['fc_units_out'] = 512
            x = fc(x,c)
            print 'fc1_shape:',x.get_shape()

        with tf.variable_scope('Value'):
            c['fc_units_out'] = 5
            v = fc(x, c)
            print 'v_state:', v.get_shape()

        with tf.variable_scope('Advantage'):
            c['fc_units_out'] = num_classes
            a = fc(x, c)
            print 'a_state', a.get_shape()

        out = v + (a - tf.reduce_mean(a, axis=1, keep_dims=True))

    return out



def inference(x, num_classes=5,
              scope_name='target_net',
              use_bias=True,  # defaults to using batch norm
             ):
    c = Config()
    c['use_bias'] = use_bias
    c['padding'] = 'SAME'

    with tf.variable_scope(scope_name):
        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 32
            c['ksize'] = 8
            c['stride'] = 4
            x = conv(x, c)
            x = activation(x)
            print "x1_shape", x.get_shape()

        with tf.variable_scope('scale2'):
            # x = _max_pool(x, ksize=2, stride=2)
            c['conv_filters_out'] = 64
            c['ksize'] = 4
            c['stride'] = 2
            x = conv(x, c)
            x = activation(x)
            print "x2_shape", x.get_shape()

        with tf.variable_scope('scale3'):
            c['conv_filters_out'] = 64
            c['ksize'] = 3
            c['stride'] = 1
            x = conv(x, c)
            x = activation(x)
            print "x3_shape", x.get_shape()

        # x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
        x = tf.reshape(x, [-1, 6400])
        with tf.variable_scope('fc1'):
            c['fc_units_out'] = 512
            x = fc(x,c)

        with tf.variable_scope('fc_out'):
            c['fc_units_out'] = num_classes
            x = fc(x, c)
        print 'fc_out',x.get_shape()
    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable(c, 'weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable(c, 'biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(c,
                  name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    var = tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=dtype,
                          regularizer=regularizer,
                          trainable=trainable)
    return var


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    try:
        a = int(filters_in)
    except:
        filters_in = 1
    shape = [ksize, ksize, filters_in, filters_out]
    std = 1 / math.sqrt(ksize * ksize * int(filters_in))

    initializer = tf.random_normal_initializer(stddev=std)
    weights = _get_variable(c, 'weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
