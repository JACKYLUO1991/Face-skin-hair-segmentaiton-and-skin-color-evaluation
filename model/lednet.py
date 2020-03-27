#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 17:42
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : lednet.py
# @Software: PyCharm

from keras import layers, models


class LEDNet:
    def __init__(self, groups, classes, input_shape):
        self.groups = groups
        self.classes = classes
        self.input_shape = input_shape

    def ss_bt(self, x, dilation, strides=(1, 1), padding='same'):
        x1, x2 = self.channel_split(x)
        filters = (int(x.shape[-1]) // self.groups)
        x1 = layers.Conv2D(filters, kernel_size=(3, 1), strides=strides, padding=padding)(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv2D(filters, kernel_size=(1, 3), strides=strides, padding=padding)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv2D(filters, kernel_size=(3, 1), strides=strides, padding=padding, dilation_rate=(dilation, 1))(
            x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv2D(filters, kernel_size=(1, 3), strides=strides, padding=padding, dilation_rate=(1, dilation))(
            x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)

        x2 = layers.Conv2D(filters, kernel_size=(1, 3), strides=strides, padding=padding)(x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.Conv2D(filters, kernel_size=(3, 1), strides=strides, padding=padding)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.Conv2D(filters, kernel_size=(1, 3), strides=strides, padding=padding, dilation_rate=(1, dilation))(
            x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.Conv2D(filters, kernel_size=(3, 1), strides=strides, padding=padding, dilation_rate=(dilation, 1))(
            x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x_concat = layers.concatenate([x1, x2], axis=-1)
        x_add = layers.add([x, x_concat])
        output = self.channel_shuffle(x_add)
        return output

    def channel_shuffle(self, x):
        n, h, w, c = x.shape.as_list()
        x_reshaped = layers.Reshape([h, w, self.groups, int(c // self.groups)])(x)
        x_transposed = layers.Permute((1, 2, 4, 3))(x_reshaped)
        output = layers.Reshape([h, w, c])(x_transposed)
        return output

    def channel_split(self, x):
        x_left = layers.Lambda(lambda y: y[:, :, :, :int(int(y.shape[-1]) // self.groups)])(x)
        x_right = layers.Lambda(lambda y: y[:, :, :, int(int(y.shape[-1]) // self.groups):])(x)
        return x_left, x_right

    def down_sample(self, x, filters):
        x_filters = int(x.shape[-1])
        x_conv = layers.Conv2D(filters - x_filters, kernel_size=3, strides=(2, 2), padding='same')(x)
        x_pool = layers.MaxPool2D()(x)
        x = layers.concatenate([x_conv, x_pool], axis=-1)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def apn_module(self, x):

        def right(x):
            x = layers.AveragePooling2D()(x)
            x = layers.Conv2D(self.classes, kernel_size=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.UpSampling2D(interpolation='bilinear')(x)
            return x

        def conv(x, filters, kernel_size, stride):
            x = layers.Conv2D(filters, kernel_size=kernel_size, strides=(stride, stride), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            return x

        x_7 = conv(x, int(x.shape[-1]), 7, stride=2)
        x_5 = conv(x_7, int(x.shape[-1]), 5, stride=2)
        x_3 = conv(x_5, int(x.shape[-1]), 3, stride=2)

        x_3_1 = conv(x_3, self.classes, 3, stride=1)
        x_3_1_up = layers.UpSampling2D(interpolation='bilinear')(x_3_1)
        x_5_1 = conv(x_5, self.classes, 5, stride=1)
        x_3_5 = layers.add([x_5_1, x_3_1_up])
        x_3_5_up = layers.UpSampling2D(interpolation='bilinear')(x_3_5)
        x_7_1 = conv(x_7, self.classes, 3, stride=1)
        x_3_5_7 = layers.add([x_7_1, x_3_5_up])
        x_3_5_7_up = layers.UpSampling2D(interpolation='bilinear')(x_3_5_7)

        x_middle = conv(x, self.classes, 1, stride=1)
        x_middle = layers.multiply([x_3_5_7_up, x_middle])

        x_right = right(x)
        x_middle = layers.add([x_middle, x_right])
        return x_middle

    def encoder(self, x):
        x = self.down_sample(x, filters=32)
        for _ in range(3):
            x = self.ss_bt(x, dilation=1)

        x = self.down_sample(x, filters=64)
        for _ in range(2):
            x = self.ss_bt(x, dilation=1)

        x = self.down_sample(x, filters=128)

        dilation_rate = [1, 2, 5, 9, 2, 5, 9, 17]
        for dilation in dilation_rate:
            x = self.ss_bt(x, dilation=dilation)
        return x

    def decoder(self, x):
        x = self.apn_module(x)
        x = layers.UpSampling2D(size=8, interpolation='bilinear')(x)
        x = layers.Conv2D(self.classes, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('softmax')(x)
        return x

    def model(self):
        inputs = layers.Input(shape=self.input_shape)
        encoder_out = self.encoder(inputs)
        output = self.decoder(encoder_out)
        return models.Model(inputs, output)


if __name__ == '__main__':
    from flops import get_flops

    model = LEDNet(2, 3, (256, 256, 3)).model()
    model.summary()

    print(get_flops(model))
