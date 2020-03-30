#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 17:43
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : mobilenet.py
# @Software: PyCharm

from keras.models import *
from keras.layers import *


def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)
    return ReLU(6, name='conv1_relu')(x)


def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_pw_%d_bn' % block_id)(x)
    return ReLU(6, name='conv_pw_%d_relu' % block_id)(x)


def MobileNet(input_shape, cls_num, alpha=0.5):
    inputs = Input(input_shape)
    x = conv_block(inputs, 16, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 16, alpha, 6, block_id=1)
    f1 = x
    x = depthwise_conv_block(x, 32, alpha, 6, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 32, alpha, 6, block_id=3)
    f2 = x
    x = depthwise_conv_block(x, 64, alpha, 6, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 64, alpha, 6, block_id=5)
    f3 = x
    x = depthwise_conv_block(x, 128, alpha, 6, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 128, alpha, 6, block_id=7)
    x = depthwise_conv_block(x, 128, alpha, 6, block_id=8)
    x = depthwise_conv_block(x, 128, alpha, 6, block_id=9)
    x = depthwise_conv_block(x, 128, alpha, 6, block_id=10)
    x = depthwise_conv_block(x, 128, alpha, 6, block_id=11)

    o = x
    o = Conv2D(128, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)
    # decode
    o = UpSampling2D((2, 2))(o)
    o = concatenate([o, f3], axis=-1)
    o = Conv2D(64, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([o, f2], axis=-1)
    o = Conv2D(32, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([o, f1], axis=-1)

    o = Conv2D(16, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)

    o = Conv2D(cls_num, (3, 3), padding='same')(o)
    o = UpSampling2D((2, 2))(o)
    o = Activation('softmax')(o)

    return Model(inputs, o)


if __name__ == '__main__':
    from flops import get_flops

    model = MobileNet(input_shape=(256, 256, 3), cls_num=3)
    model.summary()

    get_flops(model, True)
