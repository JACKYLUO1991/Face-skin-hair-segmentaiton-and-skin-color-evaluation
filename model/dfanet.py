#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 19:56
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : dfanet.py
# @Software: PyCharm

from keras.layers import *
from keras.models import Model
import keras.backend as K


def ConvBlock(inputs, n_filters, kernel_size=3, strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = Conv2D(n_filters, kernel_size, strides=strides,
                 padding='same',
                 kernel_initializer='he_normal',
                 use_bias=False)(inputs)

    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    return net


def separable_res_block_deep(inputs, nb_filters, filter_size=3, strides=1, dilation=1, ix=0):
    inputs = Activation('relu')(inputs)  # , name=prefix + '_sepconv1_act'

    ip_nb_filter = K.get_variable_shape(inputs)[-1]
    if ip_nb_filter != nb_filters or strides != 1:
        residual = Conv2D(nb_filters, 1, strides=strides, use_bias=False)(inputs)
        residual = BatchNormalization()(residual)
    else:
        residual = inputs

    x = SeparableConv2D(nb_filters // 4, filter_size,
                        dilation_rate=dilation,
                        padding='same',
                        use_bias=False,
                        kernel_initializer='he_normal',
                        )(inputs)
    x = BatchNormalization()(x)  # name=prefix + '_sepconv1_bn'

    x = Activation('relu')(x)  # , name=prefix + '_sepconv2_act'
    x = SeparableConv2D(nb_filters // 4, filter_size,
                        dilation_rate=dilation,
                        padding='same',
                        use_bias=False,
                        kernel_initializer='he_normal',
                        )(x)
    x = BatchNormalization()(x)  # name=prefix + '_sepconv2_bn'
    x = Activation('relu')(x)  # , name=prefix + '_sepconv3_act'
    # if strides != 1:
    x = SeparableConv2D(nb_filters, filter_size,
                        strides=strides,
                        dilation_rate=dilation,
                        padding='same',
                        use_bias=False,
                        )(x)

    x = BatchNormalization()(x)  # name=prefix + '_sepconv3_bn'
    x = add([x, residual])
    return x


def encoder(inputs, nb_filters, stage):
    rep_nums = 0
    if stage == 2 or stage == 4:
        rep_nums = 4
    elif stage == 3:
        rep_nums = 6
    x = separable_res_block_deep(inputs, nb_filters, strides=2)  # , ix = rand_nb + stage * 10
    for i in range(rep_nums - 1):
        x = separable_res_block_deep(x, nb_filters, strides=1)  # , ix = rand_nb + stage * 10 + i

    return x


def AttentionRefinementModule(inputs):
    # Global average pooling
    nb_channels = K.get_variable_shape(inputs)[-1]
    net = GlobalAveragePooling2D()(inputs)

    net = Reshape((1, nb_channels))(net)
    net = Conv1D(nb_channels, kernel_size=1,
                 kernel_initializer='he_normal',
                 )(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Conv1D(nb_channels, kernel_size=1,
                 kernel_initializer='he_normal',
                 )(net)
    net = BatchNormalization()(net)
    net = Activation('sigmoid')(net)  # tf.sigmoid(net)

    net = Multiply()([inputs, net])

    return net


def xception_backbone(inputs, size_factor=2):
    x = Conv2D(8, kernel_size=3, strides=2,
               padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = encoder(x, int(16 * size_factor), 2)
    x = encoder(x, int(32 * size_factor), 3)
    x = encoder(x, int(64 * size_factor), 4)

    x = AttentionRefinementModule(x)
    return x


def DFANet(input_shape, cls_num=3, size_factor=2):
    img_input = Input(input_shape)

    x = Conv2D(8, kernel_size=5, strides=2,
               padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    levela_input = Activation('relu')(x)

    enc2_a = encoder(levela_input, int(16 * size_factor), 2)

    enc3_a = encoder(enc2_a, int(32 * size_factor), 3)

    enc4_a = encoder(enc3_a, int(64 * size_factor), 4)

    enc_attend_a = AttentionRefinementModule(enc4_a)

    enc_upsample_a = UpSampling2D(size=4, interpolation='bilinear')(enc_attend_a)

    levelb_input = Concatenate()([enc2_a, enc_upsample_a])
    enc2_b = encoder(levelb_input, int(16 * size_factor), 2)

    enc2_b_combine = Concatenate()([enc3_a, enc2_b])
    enc3_b = encoder(enc2_b_combine, int(32 * size_factor), 3)

    enc3_b_combine = Concatenate()([enc4_a, enc3_b])
    enc4_b = encoder(enc3_b_combine, int(64 * size_factor), 4)

    enc_attend_b = AttentionRefinementModule(enc4_b)

    enc_upsample_b = UpSampling2D(size=4, interpolation='bilinear')(enc_attend_b)

    levelc_input = Concatenate()([enc2_b, enc_upsample_b])
    enc2_c = encoder(levelc_input, int(16 * size_factor), 2)

    enc2_c_combine = Concatenate()([enc3_b, enc2_c])
    enc3_c = encoder(enc2_c_combine, int(32 * size_factor), 3)

    enc3_c_combine = Concatenate()([enc4_b, enc3_c])
    enc4_c = encoder(enc3_c_combine, int(64 * size_factor), 4)

    enc_attend_c = AttentionRefinementModule(enc4_c)

    enc2_a_decoder = ConvBlock(enc2_a, 32, kernel_size=1)

    enc2_b_decoder = ConvBlock(enc2_b, 32, kernel_size=1)
    enc2_b_decoder = UpSampling2D(size=2, interpolation='bilinear')(enc2_b_decoder)

    enc2_c_decoder = ConvBlock(enc2_c, 32, kernel_size=1)
    enc2_c_decoder = UpSampling2D(size=4, interpolation='bilinear')(enc2_c_decoder)

    decoder_front = Add()([enc2_a_decoder, enc2_b_decoder, enc2_c_decoder])
    decoder_front = ConvBlock(decoder_front, 32, kernel_size=1)

    att_a_decoder = ConvBlock(enc_attend_a, 32, kernel_size=1)
    att_a_decoder = UpSampling2D(size=4, interpolation='bilinear')(att_a_decoder)

    att_b_decoder = ConvBlock(enc_attend_b, 32, kernel_size=1)
    att_b_decoder = UpSampling2D(size=8, interpolation='bilinear')(att_b_decoder)

    att_c_decoder = ConvBlock(enc_attend_c, 32, kernel_size=1)
    att_c_decoder = UpSampling2D(size=16, interpolation='bilinear')(att_c_decoder)

    decoder_combine = Add()([decoder_front, att_a_decoder, att_b_decoder, att_c_decoder])

    decoder_combine = ConvBlock(decoder_combine, cls_num * 2, kernel_size=1)

    decoder_final = UpSampling2D(size=4, interpolation='bilinear')(decoder_combine)
    output = Conv2D(cls_num, (1, 1), activation='softmax')(decoder_final)

    return Model(img_input, output, name='DFAnet')


if __name__ == '__main__':
    from flops import get_flops

    model = DFANet(input_shape=(256, 256, 3), cls_num=3, size_factor=2)
    model.summary()

    get_flops(model)
