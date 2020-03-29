#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 17:49
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    :
# @File    : flops.py
# @Software: PyCharm

# https://github.com/ckyrkou/Keras_FLOP_Estimator

import keras.backend as K


def get_flops(model, table=False):
    if table:
        print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('-' * 170)

    t_flops = 0
    t_macc = 0

    for l in model.layers:

        o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], [1, 1], [0, 0], [0, 0]
        flops = 0
        macc = 0
        name = l.name

        factor = 1e9

        if 'InputLayer' in str(l):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = i_shape

        if 'Reshape' in str(l):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

        if 'Add' in str(l) or 'Maximum' in str(l) or 'Concatenate' in str(l):
            i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
            o_shape = l.output.get_shape()[1:4].as_list()
            flops = (len(l.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]

        if 'Average' in str(l) and 'pool' not in str(l):
            i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
            o_shape = l.output.get_shape()[1:4].as_list()
            flops = len(l.input) * i_shape[0] * i_shape[1] * i_shape[2]

        if 'BatchNormalization' in str(l):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

            bflops = 1
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor

        if 'Activation' in str(l) or 'activation' in str(l):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            bflops = 1
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor

        if 'pool' in str(l) and ('Global' not in str(l)):
            i_shape = l.input.get_shape()[1:4].as_list()
            strides = l.strides
            ks = l.pool_size
            flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))

        if 'Flatten' in str(l):
            i_shape = l.input.shape[1:4].as_list()
            flops = 1
            out_vec = 1
            for i in range(len(i_shape)):
                flops *= i_shape[i]
                out_vec *= i_shape[i]
            o_shape = flops
            flops = 0

        if 'Dense' in str(l):
            print(l.input)
            i_shape = l.input.shape[1:4].as_list()[0]
            if i_shape is None:
                i_shape = out_vec

            o_shape = l.output.shape[1:4].as_list()
            flops = 2 * (o_shape[0] * i_shape)
            macc = flops / 2

        if 'Padding' in str(l):
            flops = 0

        if 'Global' in str(l):
            i_shape = l.input.get_shape()[1:4].as_list()
            flops = ((i_shape[0]) * (i_shape[1]) * (i_shape[2]))
            o_shape = [l.output.get_shape()[1:4].as_list(), 1, 1]
            out_vec = o_shape

        if 'Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l):
            strides = l.strides
            ks = l.kernel_size
            filters = l.filters
            # if 'Conv2DTranspose' in str(l):
            #     i_shape = list(K.int_shape(l.input)[1:4])
            #     o_shape = list(K.int_shape(l.output)[1:4])
            # else:
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

            if filters is None:
                filters = i_shape[2]

            flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                    (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
            macc = flops / 2

        if 'Conv2D' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l):
            strides = l.strides
            ks = l.kernel_size
            filters = l.filters
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

            if filters is None:
                filters = i_shape[2]

            flops = 2 * ((ks[0] * ks[1] * i_shape[2]) * ((i_shape[0] / strides[0]) * (
                    i_shape[1] / strides[1]))) / factor
            macc = flops / 2

        t_macc += macc

        t_flops += flops

        if table:
            print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                name, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
    t_flops = t_flops / factor

    print('Total FLOPS (x 10^-9): %10.8f G' % (t_flops))
    print('Total MACCs: %10.8f\n' % (t_macc))

    return
