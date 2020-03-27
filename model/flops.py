#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 17:49
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    :
# @File    : flops.py
# @Software: PyCharm

import tensorflow as tf
import keras.backend as K


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops
